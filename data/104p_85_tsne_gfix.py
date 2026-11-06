#!/usr/bin/env python3
"""第 104p 步:Fig5c t-SNE 点的 gRNAde C/G 解码修正重算(→ survey/85_tsne_points_gfix.json)

背景:85_tsne_points.json 的 gRNAde 4-mer 向量由旧解码池(grnade_out_btest_v2,
26v2 解码 A,C,G,U 写盘序与真实 A=0,G=1,C=2,U=3 不符)算出。修正池
grnade_out_btest_v2_cfix 为逐条 C↔G 互换,4-mer 组成随之改变,需重算。

口径与 85_tsne_kl.py 完全一致(唯一差异 = gRNAde 池目录):
  60 抽样链(RandomState(42),链集 = splits_v2 test ∩ paco_s1_v2, len≤512)
  × {native + 五方法 t=0.1 池前 30 条等长设计} 的 4-mer 组成向量(kmer_vec
  逐行拷贝自 83_novelty.py,避免其模块级重导入链)→ TSNE(perplexity=40,
  init="pca", random_state=0) 联合嵌入。

--validate 模式:用旧池 grnade_out_btest_v2 跑同一管线,核对
  (a) 60 链顺序与原 85_tsne_points.json 逐一致;
  (b) 非 gRNAde 点的 4-mer 向量逐一致(由构造保证,断言点数与链序即可);
  (c) 重算嵌入与原 JSON 坐标最大偏差(检验本机 sklearn 复现性)。

输出(仅默认模式写盘):survey/85_tsne_points_gfix.json,不覆盖原文件。
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
N_CHAINS_TSNE = 60
N_DESIGNS = 30
K = 4


def kmer_vec(seq):                       # 逐行拷贝自 83_novelty.py(K=4)
    v = np.zeros(256, dtype=np.float32)
    for i in range(len(seq) - K + 1):
        idx = 0
        ok = True
        for c in seq[i:i + K]:
            b = "ACGU".find(c)
            if b < 0:
                ok = False
                break
            idx = idx * 4 + b
        if ok:
            v[idx] += 1
    s = v.sum()
    return v / s if s > 0 else v


def pools_of(stem, native, n, grnade_dir):
    """85_tsne_kl.py 原 pools_of,仅 gRNAde 目录参数化。"""
    pools = {}
    p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
    if os.path.exists(p):
        v = json.load(open(p))
        pools["PACo"] = [s for s in v["pool"][:100] if len(s) == len(native)][:n]
    for m, d in (("RhoDesign", "rhodesign_out_btest_v2"), ("gRNAde", grnade_dir)):
        fp = os.path.join(SUR, d, f"{stem}.0.1.seqs")
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp)
                        if len(l.strip()) == len(native)][:n]
    for m, fp in (("NA-MPNN", os.path.join(SUR, "nampnn_out_btest_v2", stem + ".seqs")),
                  ("AlignIF", os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta"))):
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp)
                        if not l.startswith(">") and len(l.strip()) == len(native)][:n]
    return {m: [s for s in ps if set(s) <= set("ACGU")] for m, ps in pools.items()}


def build_matrix(grnade_dir):
    import torch
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    data = torch.load(os.path.join(ROOT, "data", "paco_s1_v2.pt"),
                      map_location="cpu", weights_only=False)
    stems_all = sorted(s for s in set(sp["test"]) & set(data)
                       if len(data[s]["seq"]) <= 512)
    rng = np.random.RandomState(42)
    tsne_stems = list(rng.choice(stems_all, N_CHAINS_TSNE, replace=False))

    X, labels, chain_ids = [], [], []
    for stem in tsne_stems:
        native = data[stem]["seq"]
        X.append(kmer_vec(native))
        labels.append("native")
        chain_ids.append(stem)
        for m, pool in pools_of(stem, native, N_DESIGNS, grnade_dir).items():
            for s in pool:
                X.append(kmer_vec(s))
                labels.append(m)
                chain_ids.append(stem)
    return np.array(X), labels, chain_ids, tsne_stems


def run_tsne(X):
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=40, init="pca",
               random_state=0).fit_transform(np.array(X))
    print(f"[104p] t-SNE {emb.shape} | {time.time()-t0:.0f}s", flush=True)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="用旧 gRNAde 池复现原 85_tsne_points.json 以校验管线")
    args = ap.parse_args()

    if args.validate:
        X, labels, chain_ids, stems = build_matrix("grnade_out_btest_v2")
        old = json.load(open(os.path.join(SUR, "85_tsne_points.json")))
        old_chains = []
        for c in old["chain"]:
            if c not in old_chains:
                old_chains.append(c)
        assert list(stems) == old_chains, "60 链抽样与原文件不一致!"
        assert labels == old["label"] and chain_ids == old["chain"], \
            "点序/标签与原文件不一致!"
        emb = run_tsne(X)
        d = np.max(np.abs(emb - np.array([old["x"], old["y"]]).T))
        print(f"[104p][validate] n={len(labels)} 链序/标签一致; "
              f"坐标最大偏差 = {d:.6g}")
        return

    X, labels, chain_ids, stems = build_matrix("grnade_out_btest_v2_cfix")
    emb = run_tsne(X)
    out = os.path.join(SUR, "85_tsne_points_gfix.json")
    json.dump({"x": emb[:, 0].tolist(), "y": emb[:, 1].tolist(),
               "label": labels, "chain": chain_ids,
               "cfix_note": "gRNAde 点 = grnade_out_btest_v2_cfix 池(26v2 解码 "
                            "C/G 互换修正后)4-mer 向量;其余方法向量与 "
                            "85_tsne_points.json 相同;联合 t-SNE 按 85 号原参数"
                            "(perplexity=40, init=pca, random_state=0, 链抽样 "
                            "RandomState(42))重跑,坐标整体重估(2026-10-16)。"},
              open(out, "w"))
    from collections import Counter
    print(f"[104p] wrote {out} | n={len(labels)} | {dict(Counter(labels))}")


if __name__ == "__main__":
    main()
