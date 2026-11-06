#!/usr/bin/env python3
"""104j：85 号家族 KL 的 gRNAde 行用 _cfix t=0.1 池（前 100）重算 → survey/85_family_kl_gfix.json

口径与 data/85_tsne_kl.py d-lite 完全一致：
  家族 = splits_v2 TM 簇，簇内 ≥2 成员的测试链；prof = leave-one-out 簇 4-mer 分布；
  q = 设计池 4-mer 分布（平滑 1e-4）；KL(q‖prof) 逐链 → mean + boot_ci(2000,seed=0)。
  池 = _cfix {stem}.0.1.seqs 前 100（len==native，纯 ACGU）。
t-SNE 坐标（85_tsne_points.json）不重算。
其余方法行与 random_ref 从 85_family_kl.json 原样拷贝。
环境：mpl372_repro（torch 读 s1 缓存）
"""
import os, sys, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
K = 4
OUT = os.path.join(SUR, "85_family_kl_gfix.json")


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


def boot_ci(x, n=2000, seed=0):
    r = np.random.RandomState(seed)
    m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from PACoAptamer.data_utils import load_s1
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp["test"]), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    stems_all = sorted(s for s in set(sp["test"]) & set(data) if len(data[s]["seq"]) <= 512)
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
            v += kmer_vec(s)
        v += 1e-4
        return v / v.sum()

    kl_gr = []
    for stem in fam_stems:
        native = data[stem]["seq"]
        fp = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.0.1.seqs")
        if not os.path.exists(fp):
            continue
        pool = [l.strip() for l in open(fp)
                if l.strip() and len(l.strip()) == len(native)][:100]
        pool = [s for s in pool if set(s) <= set("ACGU")]
        if len(pool) < 10:
            continue
        prof = kdist([data[x]["seq"] for x in by_cluster[cluster_of[stem]]
                      if x != stem and x in data])
        q = kdist(pool)
        kl_gr.append(float(np.sum(q * np.log(q / prof))))

    orig = json.load(open(os.path.join(SUR, "85_family_kl.json")))
    out = {"n_family_chains": orig["n_family_chains"],
           "methods": dict(orig["methods"]), "random_ref": orig["random_ref"]}
    out["methods"]["gRNAde"] = {"n": len(kl_gr),
                                "kl_mean": float(np.mean(kl_gr)) if kl_gr else None,
                                "ci95": boot_ci(kl_gr) if kl_gr else None}
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix t=0.1 池前 100 按 85 号口径重算"
                        "（2026-10-16）；其余行拷贝自 85_family_kl.json；t-SNE 坐标未重算。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    o = orig["methods"]["gRNAde"]
    n = out["methods"]["gRNAde"]
    print(f"[104j] gRNAde 家族 KL: {o['kl_mean']:.4f} {o['ci95']} -> "
          f"{n['kl_mean']:.4f} {n['ci95']} (n={n['n']})")
    print(f"       对照: PACo {orig['methods']['PACo']['kl_mean']:.4f} "
          f"RhoDesign {orig['methods']['RhoDesign']['kl_mean']:.4f} "
          f"random {orig['random_ref']['kl_mean']:.4f}")


if __name__ == "__main__":
    main()
