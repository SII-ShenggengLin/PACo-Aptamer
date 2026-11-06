#!/usr/bin/env python3
"""第 81 步：配对 Wilcoxon 显著性全套（蓝图 §五 P2-11 / Fig2c）

PACo(ens2) vs 四基线，三层评估集 × 双口径：
  B 层 test 477 / A 层 indep 235 / zs 93
  recovery_best（池内 best）与 recovery_t01_mean（t=0.1 真均值）
输出：survey/81_wilcoxon.json（Δ + 95%CI bootstrap + 单双侧 p + 胜率）

【2026-10-15 口径修正】旧版 B 层误读 metrics_PACo_v2_ens2_full.json（ens2-withiface
manifest 口径）、A 层误读 alayer_v2_retrained_ens2_withiface_20260818.json，与论文
主配置 ens2-noiface 不符（旧 JSON 备份 81_wilcoxon_withiface_backup.json）。
现改为与主表/正文同一来源：
  B 层 = btest_v2_main_table.json["per_chain"]（canonical 真值，63 号产物）
  A 层 = alayer_v2_retrained.json（69 号产物，正文 zs/indep 数字即来自此文件）
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")

MAIN_TABLE = "btest_v2_main_table.json"   # per_chain[method][stem]，canonical 口径
A_FILE = "alayer_v2_retrained.json"       # per_chain[stem][method]


def get(metric, caliber, key):
    v = metric.get(key)
    return None if v is None else v.get(caliber)


def norm_cal(m, s, cal):
    """统一口径键名：RhoDesign/gRNAde 用 recovery_best_overall / recovery_t0.1"""
    v = m.get(s, {})
    if cal == "recovery_best":
        for k in ("recovery_best", "recovery_best_overall"):
            if k in v:
                return v[k]
    else:
        for k in ("recovery_t01_mean", "recovery_t0.1"):
            if k in v:
                x = v[k]
                if isinstance(x, dict):      # RhoDesign/gRNAde：t0.1 存 {best, mean, n}
                    x = x.get("mean", x.get("best"))
                return x
    return None


def paired(a_map, b_map, caliber):
    common = sorted(set(a_map) & set(b_map))
    a = [a_map[s] for s in common]
    b = [b_map[s] for s in common]
    if len(common) < 10:
        return {"n": len(common)}
    d = np.array(a) - np.array(b)
    rng = np.random.RandomState(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
    out = {
        "n": len(common),
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "win_rate": float((d > 0).mean()),
        "paco_mean": float(np.mean(a)), "base_mean": float(np.mean(b)),
    }
    try:
        out["p_two_sided"] = float(wilcoxon(a, b).pvalue)
        out["p_one_sided_greater"] = float(wilcoxon(a, b, alternative="greater").pvalue)
    except Exception:
        out["p_two_sided"] = None
        out["p_one_sided_greater"] = None
    return out


def main():
    mt = json.load(open(os.path.join(SUR, MAIN_TABLE)))["per_chain"]
    b_pc = {"PACo": mt["PACoAptamer"], "RhoDesign": mt["RhoDesign"],
            "gRNAde": mt["gRNAde"], "NA-MPNN": mt["NA-MPNN"], "AlignIF": mt["AlignIF"]}
    a_pc = json.load(open(os.path.join(SUR, A_FILE)))["per_chain"]
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    a_views = {"indep": sorted(sp["independent_test"]), "zs": sorted(sp["zero_shot_targets"])}

    out = {"B_test_477": {}, "A_indep_235": {}, "A_zs_93": {}}
    for base in ("RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"):
        out["B_test_477"][base] = {}
        for cal in ("recovery_best", "recovery_t01_mean"):
            am = {s: norm_cal(b_pc["PACo"], s, cal) for s in b_pc["PACo"]}
            bm = {s: norm_cal(b_pc[base], s, cal) for s in b_pc[base]}
            am = {s: v for s, v in am.items() if v is not None}
            bm = {s: v for s, v in bm.items() if v is not None}
            out["B_test_477"][base][cal] = paired(am, bm, cal)
        # A 层
        for view, stems in a_views.items():
            out_view = out[f"A_{ 'indep_235' if view=='indep' else 'zs_93'}"].setdefault(base, {})
            for cal in ("recovery_best", "recovery_t01_mean"):
                am = {s: a_pc[s]["PACoAptamer"][cal] for s in stems
                      if s in a_pc and a_pc[s]["PACoAptamer"].get(cal) is not None}
                bm = {s: a_pc[s][base][cal] for s in stems
                      if s in a_pc and base in a_pc[s] and a_pc[s][base].get(cal) is not None}
                out_view[cal] = paired(am, bm, cal)

    json.dump(out, open(os.path.join(SUR, "81_wilcoxon.json"), "w"), indent=1)
    for view, bases in out.items():
        print(f"\n== {view} ==")
        for base, cals in bases.items():
            for cal, r in cals.items():
                if r.get("n", 0) < 10:
                    continue
                print(f"  PACo vs {base:<10} {cal:<18} n={r['n']:<4} "
                      f"Δ={r['delta']:+.4f} CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
                      f"胜率={r['win_rate']:.1%} p单={r['p_one_sided_greater']:.1e} p双={r['p_two_sided']:.1e}")


if __name__ == "__main__":
    main()
