#!/usr/bin/env python3
"""104h：82 号 GC/MFE 的 gRNAde 行用 _cfix t=0.1 池重算 → survey/82_gc_mfe_gfix.json

口径与 data/82_diversity_gc.py _gcmfe_worker/phase_gcmfe 一致：
  池 = _cfix {stem}.0.1.seqs（len==native 过滤）；gc_mean = 池内 (G+C)/L 均值；
  MFE = 池内唯一 ACGU 序列 RNAfold MFE/nt 的均值/中位（104e 折叠缓存取值）。
验证：C↔G 互换不改变 GC 含量 —— 断言新 gc_mean 与原值逐链一致（<1e-12）。
其余方法行与 native_ref 从 82_gc_mfe.json 原样拷贝。
环境：rna_if（纯 numpy）
"""
import os, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
CFIX_DIR = os.path.join(SUR, "grnade_out_btest_v2_cfix")
OUT = os.path.join(SUR, "82_gc_mfe_gfix.json")


def main():
    cache = json.load(open(os.path.join(SUR, "104_fold_cache_grnade_cfix.json")))
    per_chain = cache["per_chain"]

    gcs, mfes = [], []
    max_gc_diff = 0.0
    n_chains = 0
    for stem, rec in per_chain.items():
        native = rec["native"]
        fp = os.path.join(CFIX_DIR, f"{stem}.0.1.seqs")
        if not os.path.exists(fp):
            continue
        pool = [l.strip() for l in open(fp)
                if l.strip() and len(l.strip()) == len(native)]
        if not pool:
            continue
        n_chains += 1
        gc = [(s.count("G") + s.count("C")) / len(s) for s in pool]
        gc_mean = float(np.mean(gc))
        # GC 不变性验证：与未修正池同链 gc_mean 比较
        fp0 = os.path.join(SUR, "grnade_out_btest_v2", f"{stem}.0.1.seqs")
        pool0 = [l.strip() for l in open(fp0)
                 if l.strip() and len(l.strip()) == len(native)]
        gc0 = float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in pool0]))
        max_gc_diff = max(max_gc_diff, abs(gc_mean - gc0))
        gcs.append(gc_mean)
        uniq = sorted({s for s in pool if set(s) <= set("ACGU")})
        m = [rec["fold"][s][1] / len(s) for s in uniq
             if s in rec["fold"] and rec["fold"][s][1] is not None]
        if m:
            mfes.append(float(np.mean(m)))

    assert max_gc_diff < 1e-12, f"GC 不变性被破坏：max diff {max_gc_diff}"
    grow = {"gc_mean": float(np.mean(gcs)), "n": n_chains,
            "mfe_nt_mean": float(np.mean(mfes)) if mfes else None,
            "mfe_nt_median": float(np.median(mfes)) if mfes else None}

    orig = json.load(open(os.path.join(SUR, "82_gc_mfe.json")))
    out = {"methods": dict(orig["methods"]), "native_ref": orig["native_ref"]}
    out["methods"]["gRNAde"] = grow
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix t=0.1 池按 82 号口径重算"
                        "（2026-10-16，104e 折叠缓存）；GC 含量 C↔G 互换不变"
                        f"（逐链验证 max|Δ|={max_gc_diff:.2e}）；其余行拷贝自 82_gc_mfe.json。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    o = orig["methods"]["gRNAde"]
    print(f"[104h] gRNAde: gc {o['gc_mean']:.6f}->{grow['gc_mean']:.6f}（不变，已断言） "
          f"mfe/nt {o['mfe_nt_mean']:.4f}->{grow['mfe_nt_mean']:.4f} "
          f"med {o['mfe_nt_median']:.4f}->{grow['mfe_nt_median']:.4f} (n={grow['n']})")


if __name__ == "__main__":
    main()
