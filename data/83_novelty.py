#!/usr/bin/env python3
"""第 83 步：Fig5b 新颖性分析（蓝图 §五 P1-7）——设计与训练库/已知适配体的最大序列相似度

对象：B 层 test 477 各方法 t=0.1 池（PACo ens2 / RhoDesign / gRNAde / NA-MPNN / AlignIF）
参照库：
  train      splits_v2 train 5,773 链天然序列（防泄漏检验：test 链本身按簇隔离于 train）
  aptadb     AptaDB 254 条 RNA 适配体（79 号清洗产物）
算法：4-mer 组成向量预筛（cos 相似 top-60 候选）→ edlib 全局对齐（Hamming 距离/长度，
       只对同长度；长度差>5 的候选记 identity≤(1-差/长) 上界后跳过）
输出：survey/83_novelty.json（每方法：max-identity 分布分位数 + 距最近训练链的链级汇总
       + rank-ordered 曲线数据）
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m77 = _imp("m77", os.path.join(ROOT, "data", "77_p0234_analysis.py"))
N_PER_TEMP = 100
K = 4
TOP_CAND = 60


def kmer_vec(seq):
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


def build_lib(seqs):
    """(seqs, 矩阵) — L2 归一化的 4-mer 组成"""
    M = np.stack([kmer_vec(s) for s in seqs])
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return seqs, M / n


def max_identity(seq, lib_seqs, lib_M, vec):
    """预筛 top 候选 → edlib 精算最大 identity。
    长度差 >5 的候选直接跳过（曾试过按 1−|Δlen|/max 记"上界"，但那会把未验证的
    长度差上界当成达成相似度、系统性夸大指标，2026-08-19 当日回退）。"""
    import edlib
    sims = lib_M @ vec
    order = np.argsort(-sims)[:TOP_CAND]
    best = 0.0
    L = len(seq)
    for j in order:
        t = lib_seqs[j]
        if abs(len(t) - L) > 5:
            continue
        # edlib global（NW 距离含 indel）→ identity = 1 - dist/max(len)
        d = edlib.align(seq, t, task="distance", mode="NW")["editDistance"]
        ident = 1 - d / max(L, len(t))
        if ident > best:
            best = ident
    return best


def _worker(args):
    stem, native = args
    pools = {}
    p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
    if os.path.exists(p):
        v = json.load(open(p))
        pools["PACo"] = [s for s in v["pool"][:N_PER_TEMP] if len(s) == len(native)]
    for m, d in (("RhoDesign", "rhodesign_out_btest_v2"), ("gRNAde", "grnade_out_btest_v2")):
        fp = os.path.join(SUR, d, f"{stem}.0.1.seqs")
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp) if len(l.strip()) == len(native)]
    for m, fp in (("NA-MPNN", os.path.join(SUR, "nampnn_out_btest_v2", stem + ".seqs")),
                  ("AlignIF", os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta"))):
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp)
                        if not l.startswith(">") and len(l.strip()) == len(native)]
    out = {}
    for m, pool in pools.items():
        pool = [s for s in pool if set(s) <= set("ACGU")]
        if not pool:
            continue
        ids_train, ids_aptadb = [], []
        for s in pool:
            v4 = kmer_vec(s)
            n4 = np.linalg.norm(v4) or 1
            ids_train.append(max_identity(s, LIB_T_SEQS, LIB_T_M, v4 / n4))
            ids_aptadb.append(max_identity(s, LIB_A_SEQS, LIB_A_M, v4 / n4))
        out[m] = {"max_id_train": ids_train, "max_id_aptadb": ids_aptadb}
    return stem, out


# 模块级库（fork 共享）
sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
_all = json.load(open(os.path.join(ROOT, "data", "splits_v2", "seqs_all.json")))
LIB_T_SEQS, LIB_T_M = build_lib([_all[s] for s in sp["train"] if s in _all])
_apt = json.load(open(os.path.join(SUR, "79_aptadb_clean.json")))["entries"]
_apt_seqs = sorted({e["seq"] for e in _apt})
LIB_A_SEQS, LIB_A_M = build_lib(_apt_seqs)


def main():
    from PACoAptamer.data_utils import load_s1
    stems = m77.test_stems()
    data = load_s1(set(stems), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    args = [(s, data[s]["seq"]) for s in sorted(set(stems) & set(data))
            if len(data[s]["seq"]) <= 512]
    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, rec) in enumerate(ex.map(_worker, args, chunksize=2)):
            per_chain[stem] = rec
            if (i + 1) % 50 == 0:
                print(f"[83] {i+1}/{len(args)} | {time.time()-t0:.0f}s", flush=True)
    out = {"methods": {}}
    for m in ("PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"):
        tr = [x for v in per_chain.values() if m in v for x in v[m]["max_id_train"]]
        ap = [x for v in per_chain.values() if m in v for x in v[m]["max_id_aptadb"]]
        tr_s, ap_s = sorted(tr), sorted(ap)
        out["methods"][m] = {
            "n_designs": len(tr),
            "train": {"mean": float(np.mean(tr)), "median": float(np.median(tr)),
                      "q90": float(np.quantile(tr, .9)), "q99": float(np.quantile(tr, .99)),
                      "max": float(np.max(tr)),
                      "frac_gt_0.8": float(np.mean(np.array(tr) > 0.8)),
                      "rank_curve": tr_s[::max(1, len(tr_s) // 500)]},
            "aptadb": {"mean": float(np.mean(ap)), "median": float(np.median(ap)),
                       "q90": float(np.quantile(ap, .9)),
                       "frac_gt_0.8": float(np.mean(np.array(ap) > 0.8))}}
    json.dump(out, open(os.path.join(SUR, "83_novelty.json"), "w"), indent=1)
    for m, v in out["methods"].items():
        print(f"[83] {m}: vs train 中位 {v['train']['median']:.3f} q90 {v['train']['q90']:.3f} "
              f"max {v['train']['max']:.3f} >0.8 占比 {v['train']['frac_gt_0.8']:.1%} | "
              f"vs AptaDB 中位 {v['aptadb']['median']:.3f}")


if __name__ == "__main__":
    main()
