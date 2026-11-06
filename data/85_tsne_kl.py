#!/usr/bin/env python3
"""第 85 步：Fig5c 生成池结构空间 t-SNE + Fig5d-lite 家族组成保真 KL（蓝图 P1-7/P2-13）

c：60 抽样链 × {native + 五方法 t01 池前 30 设计} 的 4-mer 组成向量 → t-SNE 二维嵌入
   （坐标+标签存 JSON 供绘图；native 星标）
d-lite：Rfam 不可得 → 用 splits_v2 的 TM 簇做"家族"：对簇内 ≥2 成员的测试链，
   KL(设计池 4-mer 分布 ‖ leave-one-out 家族 profile)；随机序列对照同口径
输出：survey/85_tsne_points.json、survey/85_family_kl.json
"""
import os, sys, json, time
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m83 = _imp("m83", os.path.join(ROOT, "data", "83_novelty.py"))   # 复用 kmer_vec
N_CHAINS_TSNE = 60
N_DESIGNS = 30
METHODS = ("PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF")


def pools_of(stem, native, n):
    pools = {}
    p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
    if os.path.exists(p):
        v = json.load(open(p))
        pools["PACo"] = [s for s in v["pool"][:100] if len(s) == len(native)][:n]
    for m, d in (("RhoDesign", "rhodesign_out_btest_v2"), ("gRNAde", "grnade_out_btest_v2")):
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


def main():
    from PACoAptamer.data_utils import load_s1
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp["test"]), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    stems_all = sorted(s for s in set(sp["test"]) & set(data) if len(data[s]["seq"]) <= 512)
    rng = np.random.RandomState(42)
    tsne_stems = list(rng.choice(stems_all, N_CHAINS_TSNE, replace=False))

    # ---- c: t-SNE
    X, labels, chain_ids = [], [], []
    for stem in tsne_stems:
        native = data[stem]["seq"]
        X.append(m83.kmer_vec(native))
        labels.append("native")
        chain_ids.append(stem)
        for m, pool in pools_of(stem, native, N_DESIGNS).items():
            for s in pool:
                X.append(m83.kmer_vec(s))
                labels.append(m)
                chain_ids.append(stem)
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=40, init="pca",
               random_state=0).fit_transform(np.array(X))
    print(f"[85] t-SNE {emb.shape} | {time.time()-t0:.0f}s", flush=True)
    json.dump({"x": emb[:, 0].tolist(), "y": emb[:, 1].tolist(),
               "label": labels, "chain": chain_ids},
              open(os.path.join(SUR, "85_tsne_points.json"), "w"))

    # ---- d-lite: 家族 KL（LOO cluster profile）
    cluster_of = sp["cluster_of"]
    by_cluster = {}
    for s in stems_all:
        by_cluster.setdefault(cluster_of.get(s), []).append(s)
    fam_stems = [s for s in stems_all
                 if len([x for x in by_cluster[cluster_of.get(s)]
                         if x != s and x in data]) >= 1]

    def kdist(seqs):
        v = np.zeros(256, dtype=np.float64)
        for s in seqs:
            v += m83.kmer_vec(s)
        v += 1e-4                      # 平滑
        return v / v.sum()

    kl_out = {m: [] for m in METHODS}
    kl_rand = []
    rng2 = np.random.RandomState(7)
    for stem in fam_stems:
        native = data[stem]["seq"]
        prof = kdist([data[x]["seq"] for x in by_cluster[cluster_of[stem]]
                      if x != stem and x in data])
        pools = pools_of(stem, native, 100)
        for m in METHODS:
            if m in pools and len(pools[m]) >= 10:
                q = kdist(pools[m])
                kl_out[m].append(float(np.sum(q * np.log(q / prof))))
        rand = ["".join(rng2.choice(list("ACGU"), len(native))) for _ in range(50)]
        q = kdist(rand)
        kl_rand.append(float(np.sum(q * np.log(q / prof))))

    def boot_ci(x, n=2000, seed=0):
        r = np.random.RandomState(seed)
        m = [r.choice(x, len(x)).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    out = {"n_family_chains": len(fam_stems), "methods": {}}
    for m in METHODS:
        out["methods"][m] = {"n": len(kl_out[m]),
                             "kl_mean": float(np.mean(kl_out[m])) if kl_out[m] else None,
                             "ci95": boot_ci(kl_out[m]) if kl_out[m] else None}
    out["random_ref"] = {"n": len(kl_rand), "kl_mean": float(np.mean(kl_rand)),
                         "ci95": boot_ci(kl_rand)}
    json.dump(out, open(os.path.join(SUR, "85_family_kl.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
