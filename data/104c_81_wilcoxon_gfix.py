#!/usr/bin/env python3
"""104c：81 号配对 Wilcoxon 的 PACo-vs-gRNAde 三组用修正值重算 → survey/81_wilcoxon_gfix.json

PACo 逐链值：B 层 = btest_v2_main_table.json per_chain["PACoAptamer"]；
             A 层 = alayer_v2_retrained.json per_chain[*]["PACoAptamer"]（均不变）。
gRNAde 逐链值：B 层 = 63_main_table_gfix.json per_chain["gRNAde"]（104a 产物）；
               A 层 = 69_alayer_gfix.json per_chain[*]["gRNAde"]（104b 产物）。
统计口径与 data/81_wilcoxon_full.py 的 paired() 完全一致
（Δ mean、3000 次 bootstrap CI95 seed=0、win_rate、双侧/单侧 wilcoxon）。
其余基线行从 81_wilcoxon.json 原样拷贝（便于对照表完整）。
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def paired(a_map, b_map):
    """81 号 paired() 逐字口径"""
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
    mt = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))["per_chain"]
    g63 = json.load(open(os.path.join(SUR, "63_main_table_gfix.json")))["per_chain"]["gRNAde"]
    a_orig = json.load(open(os.path.join(SUR, "alayer_v2_retrained.json")))["per_chain"]
    a_g69 = json.load(open(os.path.join(SUR, "69_alayer_gfix.json")))["per_chain"]
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    a_views = {"A_indep_235": sorted(sp["independent_test"]),
               "A_zs_93": sorted(sp["zero_shot_targets"])}

    orig = json.load(open(os.path.join(SUR, "81_wilcoxon.json")))
    out = {v: {b: dict(c) for b, c in d.items()} for v, d in orig.items()}

    # B 层 477
    for cal in ("recovery_best", "recovery_t01_mean"):
        am = {s: v[cal] for s, v in mt["PACoAptamer"].items() if v.get(cal) is not None}
        bm = {s: v[cal] for s, v in g63.items() if v.get(cal) is not None}
        out["B_test_477"]["gRNAde"][cal] = paired(am, bm)
    # A 层两视图
    for view, stems in a_views.items():
        for cal in ("recovery_best", "recovery_t01_mean"):
            am = {s: a_orig[s]["PACoAptamer"][cal] for s in stems
                  if s in a_orig and a_orig[s]["PACoAptamer"].get(cal) is not None}
            bm = {s: a_g69[s]["gRNAde"][cal] for s in stems
                  if s in a_g69 and "gRNAde" in a_g69[s]
                  and a_g69[s]["gRNAde"].get(cal) is not None}
            out[view]["gRNAde"][cal] = paired(am, bm)

    out["cfix_note"] = ("gRNAde 列 = C/G 解码修正后重算（104a/104b 产物，2026-10-16）；"
                        "PACo 与其余基线行拷贝自 81_wilcoxon.json。")
    atomic_json(out, os.path.join(SUR, "81_wilcoxon_gfix.json"))
    for view in ("B_test_477", "A_indep_235", "A_zs_93"):
        for cal in ("recovery_best", "recovery_t01_mean"):
            o = orig[view]["gRNAde"][cal]
            n = out[view]["gRNAde"][cal]
            print(f"[104c] {view} {cal}: Δ {o['delta']:+.4f}->{n['delta']:+.4f} "
                  f"胜率 {o['win_rate']:.3f}->{n['win_rate']:.3f} "
                  f"p单 {o['p_one_sided_greater']:.2e}->{n['p_one_sided_greater']:.2e}")


if __name__ == "__main__":
    main()
