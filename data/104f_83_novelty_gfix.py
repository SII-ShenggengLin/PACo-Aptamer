#!/usr/bin/env python3
"""104f：83 号 novelty 的 gRNAde 行用 _cfix t=0.1 池重算 → survey/83_novelty_gfix.json

算法与 data/83_novelty.py 完全一致（4-mer 组成向量 cos 预筛 top-60 → edlib NW
全局距离，identity = 1 - dist/max(len)，长度差>5 候选跳过），仅把
grnade_out_btest_v2/{stem}.0.1.seqs 换成 _cfix 版本。
其余方法行从 survey/83_novelty.json 原样拷贝。
环境：mpl372_repro + PYTHONPATH=/tmp/edlib_lib（edlib）
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
K = 4
TOP_CAND = 60
OUT = os.path.join(SUR, "83_novelty_gfix.json")


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
    M = np.stack([kmer_vec(s) for s in seqs])
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1
    return seqs, M / n


def max_identity(seq, lib_seqs, lib_M, vec):
    import edlib
    sims = lib_M @ vec
    order = np.argsort(-sims)[:TOP_CAND]
    best = 0.0
    L = len(seq)
    for j in order:
        t = lib_seqs[j]
        if abs(len(t) - L) > 5:
            continue
        d = edlib.align(seq, t, task="distance", mode="NW")["editDistance"]
        ident = 1 - d / max(L, len(t))
        if ident > best:
            best = ident
    return best


def _worker(args):
    stem, native = args
    fp = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.0.1.seqs")
    if not os.path.exists(fp):
        return stem, None
    pool = [l.strip() for l in open(fp)
            if l.strip() and len(l.strip()) == len(native)]
    pool = [s for s in pool if set(s) <= set("ACGU")]
    if not pool:
        return stem, None
    ids_train, ids_aptadb = [], []
    for s in pool:
        v4 = kmer_vec(s)
        n4 = np.linalg.norm(v4) or 1
        ids_train.append(max_identity(s, LIB_T_SEQS, LIB_T_M, v4 / n4))
        ids_aptadb.append(max_identity(s, LIB_A_SEQS, LIB_A_M, v4 / n4))
    return stem, {"max_id_train": ids_train, "max_id_aptadb": ids_aptadb}


# 模块级库（fork 共享，与 83 号相同）
sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
_all = json.load(open(os.path.join(ROOT, "data", "splits_v2", "seqs_all.json")))
LIB_T_SEQS, LIB_T_M = build_lib([_all[s] for s in sp["train"] if s in _all])
_apt = json.load(open(os.path.join(SUR, "79_aptadb_clean.json")))["entries"]
_apt_seqs = sorted({e["seq"] for e in _apt})
LIB_A_SEQS, LIB_A_M = build_lib(_apt_seqs)


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from PACoAptamer.data_utils import load_s1
    stems = list(sp["test"])
    data = load_s1(set(stems), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    args = [(s, data[s]["seq"]) for s in sorted(set(stems) & set(data))
            if len(data[s]["seq"]) <= 512]
    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, rec) in enumerate(ex.map(_worker, args, chunksize=2)):
            if rec is not None:
                per_chain[stem] = rec
            if (i + 1) % 50 == 0:
                print(f"[104f] {i+1}/{len(args)} | {time.time()-t0:.0f}s", flush=True)

    tr = [x for v in per_chain.values() for x in v["max_id_train"]]
    ap = [x for v in per_chain.values() for x in v["max_id_aptadb"]]
    tr_s = sorted(tr)
    grow = {
        "n_designs": len(tr),
        "train": {"mean": float(np.mean(tr)), "median": float(np.median(tr)),
                  "q90": float(np.quantile(tr, .9)), "q99": float(np.quantile(tr, .99)),
                  "max": float(np.max(tr)),
                  "frac_gt_0.8": float(np.mean(np.array(tr) > 0.8)),
                  "rank_curve": tr_s[::max(1, len(tr_s) // 500)]},
        "aptadb": {"mean": float(np.mean(ap)), "median": float(np.median(ap)),
                   "q90": float(np.quantile(ap, .9)),
                   "frac_gt_0.8": float(np.mean(np.array(ap) > 0.8))},
    }
    orig = json.load(open(os.path.join(SUR, "83_novelty.json")))
    out = {"methods": dict(orig["methods"])}
    out["methods"]["gRNAde"] = grow
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix t=0.1 池按 83 号口径重算"
                        "（2026-10-16）；其余方法行拷贝自 83_novelty.json。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    o = orig["methods"]["gRNAde"]
    print(f"[104f] gRNAde vs train: 中位 {o['train']['median']:.3f}->{grow['train']['median']:.3f} "
          f"q90 {o['train']['q90']:.3f}->{grow['train']['q90']:.3f} "
          f">0.8 {o['train']['frac_gt_0.8']:.3f}->{grow['train']['frac_gt_0.8']:.3f}")
    print(f"       vs AptaDB: 中位 {o['aptadb']['median']:.3f}->{grow['aptadb']['median']:.3f}")
    print(f"       总耗时 {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
