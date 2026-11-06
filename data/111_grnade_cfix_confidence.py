#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""111b: gRNAde cfix 池置信度指标 —— 严格按 107_conf_cross_baselines 口径
(逐样本 recovery=N 位跳过逐位一致率; 链级真值=63_main_table_gfix.json gRNAde t01;
 链间 Spearman + bootstrap 1000× seed 20261015; 链内 ≥30 样本、degenerate/恒定 recovery 剔除;
 top-10% 富集=round(0.1n) 按 logp 降序)。
输入: survey/111_grnade_cfix_logp.json (111 官方 vocab 重打分)
输出: survey/111_grnade_cfix_confidence.json
"""
import os, json
import numpy as np
from scipy.stats import spearmanr

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
BOOT_N, BOOT_SEED = 1000, 20261015
MIN_WITHIN = 30
DEGEN_MIN_UNIQUE, DEGEN_MIN_VAR = 5, 1e-10

canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
test = sorted(json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))["test"])
gfix_pc = json.load(open(os.path.join(SUR, "63_main_table_gfix.json")))["per_chain"]["gRNAde"]
p111 = json.load(open(os.path.join(SUR, "111_grnade_cfix_logp.json")))["per_chain"]


def read(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def topk_enrich(logp, rec, frac=0.1):
    n = len(logp)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(logp)[::-1][:k]
    return float(np.mean([rec[i] for i in idx]) - np.mean(rec))


def boot_ci(vals, fn=np.mean):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(vals)
    bs = [fn(vals[rng.integers(0, n, n)]) for _ in range(BOOT_N)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_spearman_ci(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(x)
    bs = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        bs.append(spearmanr(x[idx], y[idx]).statistic)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def is_degenerate(logp):
    a = np.asarray(logp, dtype=float)
    return len(np.unique(a)) < DEGEN_MIN_UNIQUE or float(np.var(a)) < DEGEN_MIN_VAR


def within_block(rhos):
    rhos = np.asarray(rhos, dtype=float)
    return {
        "min_samples_per_chain": MIN_WITHIN,
        "n_chains_valid": int(len(rhos)),
        "mean": float(rhos.mean()),
        "median": float(np.median(rhos)),
        "ci95": boot_ci(rhos),
        "frac_positive": float((rhos > 0).mean()),
    }


pc, ml, rt, wr, te = {}, [], [], [], []
degen, const_rec, skipped = [], [], {}
rec_cons_max = 0.0
for stem in test:
    e = p111.get(stem, {})
    logp = e.get("sample_logp_t01")
    if not logp:
        skipped[stem] = "no sample_logp_t01"
        continue
    L = e["len"]
    pool = [s for s in read(os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.0.1.seqs"))
            if len(s) == L]
    assert len(pool) == len(logp), f"{stem}: 池与 logp 数不齐"
    native = canon[stem]["seq"]
    npos = [i for i, c in enumerate(native) if c != "N"]
    if L != len(native):
        skipped[stem] = f"len mismatch: 111={L} canon={len(native)}"
        continue
    rec = np.array([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in pool])
    logp = np.asarray(logp)
    mt = gfix_pc[stem]
    rec_cons_max = max(rec_cons_max, abs(float(rec.mean()) - mt["recovery_t01_mean"]))
    deg = is_degenerate(logp)
    crec = len(np.unique(rec)) < 2
    rho = None
    if len(logp) >= MIN_WITHIN and not deg and not crec:
        rho = float(spearmanr(logp, rec).statistic)
        wr.append(rho)
    if deg:
        degen.append(stem)
    if crec:
        const_rec.append(stem)
    enr = topk_enrich(logp, rec)
    te.append(enr)
    pc[stem] = {"n_samples": int(len(logp)), "mean_logp": float(logp.mean()),
                "logp_var": float(np.var(logp)), "logp_n_unique": int(len(np.unique(logp))),
                "degenerate": bool(deg), "constant_recovery": bool(crec),
                "recovery_t01_mean_maintable": mt["recovery_t01_mean"],
                "recovery_t01_mean_recomputed": float(rec.mean()),
                "within_rho": rho, "top10pct_enrichment": enr}
    ml.append(logp.mean()); rt.append(mt["recovery_t01_mean"])

out = {
    "meta": {
        "step": "111_grnade_cfix_confidence",
        "date": "2026-10-19",
        "purpose": "补 107 号 gRNAde 置信度 n/a 行(官方 vocab 重打分后)",
        "convention": "与 107_conf_cross_baselines 完全一致(同 bootstrap 种子/次数、同剔除规则、"
                      "同 top-10% 定义);逐样本 logp 改自 111_grnade_cfix_logp.json;"
                      "链级真值=63_main_table_gfix.json gRNAde.recovery_t01_mean",
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "method": "percentile, resample chains"},
        "logp_source": "survey/111_grnade_cfix_logp.json(官方 L2N={A:0,G:1,C:2,U:3},cfix 池)",
        "pool": "survey/grnade_out_btest_v2_cfix/{stem}.0.1.seqs",
    },
    "n_chains": len(pc),
    "skipped_chains": skipped,
    "recovery_consistency_vs_maintable_maxabs": rec_cons_max,
    "chain_level_spearman_t01": {
        "rho": float(spearmanr(ml, rt).statistic),
        "ci95": boot_spearman_ci(ml, rt),
        "n_chains": len(ml),
    },
    "within_chain_spearman": within_block(wr),
    "within_degenerate": {
        "n_degenerate": len(degen), "degenerate_chains": degen,
        "n_constant_recovery": len(const_rec), "constant_recovery_chains": const_rec,
    },
    "top10pct_enrichment": {"mean": float(np.mean(te)), "ci95": boot_ci(te),
                            "n_chains": len(te)},
    "per_chain": pc,
}
fp = os.path.join(SUR, "111_grnade_cfix_confidence.json")
json.dump(out, open(fp + ".tmp", "w"), ensure_ascii=False)
os.rename(fp + ".tmp", fp)
cl = out["chain_level_spearman_t01"]
w = out["within_chain_spearman"]
print(f"chains={len(pc)} skipped={len(skipped)} rec_cons_max={rec_cons_max:.2e}")
print(f"chain-level rho = {cl['rho']:.4f} [{cl['ci95'][0]:.4f}, {cl['ci95'][1]:.4f}] (n={cl['n_chains']})")
print(f"within mean = {w['mean']:.4f} median {w['median']:.4f} [{w['ci95'][0]:.4f}, {w['ci95'][1]:.4f}] "
      f"frac>0 {w['frac_positive']:.3f} (n={w['n_chains_valid']}; degen {len(degen)}, const-rec {len(const_rec)})")
print(f"top10% enrich = {out['top10pct_enrichment']['mean']:+.5f} "
      f"[{out['top10pct_enrichment']['ci95'][0]:+.5f}, {out['top10pct_enrichment']['ci95'][1]:+.5f}]")
