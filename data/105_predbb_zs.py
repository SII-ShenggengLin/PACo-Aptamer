#!/usr/bin/env python3
"""第 105 步：预测骨架 × zero-shot 交叉切片（补充实验 S1，纯 CPU）

问题：预测骨架基准（193 条 cluster-based test 链，晶体骨架 → RhoFold+ 预测骨架）
与 zero-shot 层（93 链，配体训练期未出现）的交集上，PACo 对 RhoDesign 的领先优势
是否保持？

输入（均为 data/survey/ 既有 JSON，只读）：
  - 预测骨架逐链：94_predbb_paco / 101_rhodesign_predbb / 103_grnade_predbb /
    102_nampnn_predbb / 100_alignif_predbb（各 193 链 per_chain，
    recovery_best / recovery_t01_mean）
  - zero-shot 93 链清单：95_lofo.json zs_table.*.vs_baseline.per_chain_delta 键
    （与 93_tanimoto_zs.json per_chain 键完全一致，已校验）
  - 晶体骨架对照：btest_v2_main_table.json per_chain[method][chain]（477 链口径）

输出：survey/105_predbb_zs.json（临时文件 + rename）
  meta / sets（交集差集数量校验）/ validation（193 链全量均值对参考值）/
  zs_subset / non_zs_subset / full193 三口径汇总 /
  zs_subset_crystal_decomp（同链晶体口径 + Δ(predbb−crystal) 分解）/ per_chain 明细
环境：系统 python3（numpy + scipy）
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
OUT_TMP = os.path.join(SUR, "105_predbb_zs.json.tmp")
OUT = os.path.join(SUR, "105_predbb_zs.json")

METHODS = ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
PREDBB_FILES = {
    "PACoAptamer": "94_predbb_paco.json",
    "RhoDesign": "101_rhodesign_predbb.json",
    "gRNAde": "103_grnade_predbb.json",
    "NA-MPNN": "102_nampnn_predbb.json",
    "AlignIF": "100_alignif_predbb.json",
}
# 参考校验值（best/t01 全 193 链均值，百分数）
REF_FULL193 = {
    "PACoAptamer": (68.95, 51.81),
    "RhoDesign": (61.36, 47.99),
    "gRNAde": (50.66, 35.00),
    "NA-MPNN": (35.28, 29.64),
    "AlignIF": (41.89, 36.80),
}
WILCOXON_MIN_N = 10  # 与 93_tanimoto_zs.py 的 wilc 一致：n<10 不做 Wilcoxon


def boot_ci(x, n=2000, seed=0):
    """项目惯例 bootstrap 95% CI（同 104i_77p02_gfix.py）"""
    x = list(x)
    if len(x) < 3:
        return None
    rng = np.random.RandomState(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def desc(vals):
    """描述统计：mean / median / ci95"""
    vals = [float(v) for v in vals]
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "median": float(np.median(vals)), "ci95": boot_ci(vals)}


def paired(diffs):
    """配对差值统计：mean / median / ci95 / Wilcoxon 双侧 p（n>=10 才做）"""
    diffs = [float(d) for d in diffs]
    row = {"n": len(diffs), "mean": float(np.mean(diffs)),
           "median": float(np.median(diffs)), "ci95": boot_ci(diffs),
           "wilcoxon_p_two_sided": None}
    if len(diffs) >= WILCOXON_MIN_N:
        try:
            row["wilcoxon_p_two_sided"] = float(wilcoxon(diffs).pvalue)
        except Exception:
            row["wilcoxon_p_two_sided"] = None
    return row


def subset_summary(chains, predbb):
    """一个链集合上：五方法 predbb best/t01 描述统计 + PACo−RhoDesign 配对 Δ"""
    out = {"n_chains": len(chains), "chains": sorted(chains), "methods": {}}
    for m in METHODS:
        pc = predbb[m]
        out["methods"][m] = {
            "best": desc(pc[c]["recovery_best"] for c in chains),
            "t01": desc(pc[c]["recovery_t01_mean"] for c in chains),
        }
    out["delta_PACo_minus_RhoDesign"] = {
        "best": paired(predbb["PACoAptamer"][c]["recovery_best"]
                       - predbb["RhoDesign"][c]["recovery_best"] for c in chains),
        "t01": paired(predbb["PACoAptamer"][c]["recovery_t01_mean"]
                      - predbb["RhoDesign"][c]["recovery_t01_mean"] for c in chains),
    }
    return out


def main():
    # ---- 读入
    predbb = {}
    for m, fn in PREDBB_FILES.items():
        d = json.load(open(os.path.join(SUR, fn)))
        predbb[m] = d["per_chain"]
    lofo = json.load(open(os.path.join(SUR, "95_lofo.json")))
    t93 = json.load(open(os.path.join(SUR, "93_tanimoto_zs.json")))
    maintab = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))["per_chain"]

    # ---- 链集合与校验
    predbb_sets = {m: set(pc) for m, pc in predbb.items()}
    base = predbb_sets["PACoAptamer"]
    five_identical = all(s == base for s in predbb_sets.values())
    zs_lofo = set(lofo["zs_table"]["s2_v2_lofo_aminoglycoside"]
                  ["vs_baseline"]["per_chain_delta"])
    zs_93 = set(t93["per_chain"])
    assert zs_lofo == zs_93, "两个来源的 zs 93 链清单不一致"
    zs = zs_93
    overlap = sorted(base & zs)
    non_zs = sorted(base - zs)
    zs_not_in_predbb = sorted(zs - base)
    in_maintab = all(all(c in maintab[m] for c in base) for m in METHODS)

    sets_block = {
        "n_predbb": len(base), "n_zs": len(zs),
        "n_overlap": len(overlap), "n_predbb_non_zs": len(non_zs),
        "n_zs_not_in_predbb": len(zs_not_in_predbb),
        "checks": {
            "five_predbb_chain_sets_identical": five_identical,
            "zs_list_lofo_eq_tanimoto93": True,
            "overlap_plus_non_zs_eq_predbb": len(overlap) + len(non_zs) == len(base),
            "all_predbb_chains_in_main_table": in_maintab,
        },
        "overlap_chains": overlap,
        "zs_not_in_predbb_chains": zs_not_in_predbb,
    }
    print(f"predbb {len(base)} ∩ zs {len(zs)} = {len(overlap)}；"
          f"非 zs {len(non_zs)}；zs 不在 predbb {len(zs_not_in_predbb)}")

    # ---- validation：全 193 均值对参考值
    validation = {"reference_full193_best_t01_pct": REF_FULL193, "methods": {}}
    ok_all = True
    for m in METHODS:
        b = 100 * np.mean([predbb[m][c]["recovery_best"] for c in base])
        t = 100 * np.mean([predbb[m][c]["recovery_t01_mean"] for c in base])
        rb, rt = REF_FULL193[m]
        ok = abs(b - rb) < 0.01 and abs(t - rt) < 0.01
        ok_all &= ok
        validation["methods"][m] = {"best_mean_pct": round(float(b), 4),
                                    "t01_mean_pct": round(float(t), 4),
                                    "matches_reference": bool(ok)}
    validation["all_match"] = bool(ok_all)
    print("validation all_match:", ok_all)

    # ---- 三口径汇总
    zs_sum = subset_summary(overlap, predbb)
    nz_sum = subset_summary(non_zs, predbb)
    full_sum = subset_summary(sorted(base), predbb)
    for tag, s in [("zs", zs_sum), ("non-zs", nz_sum), ("full", full_sum)]:
        d = s["delta_PACo_minus_RhoDesign"]
        print(f"[{tag}] n={s['n_chains']} "
              f"PACo best={s['methods']['PACoAptamer']['best']['mean']:.4f} "
              f"Rho best={s['methods']['RhoDesign']['best']['mean']:.4f} "
              f"Δbest={d['best']['mean']:+.4f} (p={d['best']['wilcoxon_p_two_sided']:.4g}) "
              f"Δt01={d['t01']['mean']:+.4f} (p={d['t01']['wilcoxon_p_two_sided']:.4g})")

    # ---- zs 子集的晶体口径 + Δ(predbb−crystal) 分解
    cry = {m: {c: maintab[m][c] for c in overlap} for m in METHODS}
    cry_sum = {"n_chains": len(overlap), "methods": {}}
    for m in METHODS:
        cry_sum["methods"][m] = {
            "best": desc(cry[m][c]["recovery_best"] for c in overlap),
            "t01": desc(cry[m][c]["recovery_t01_mean"] for c in overlap),
        }
    cry_sum["delta_PACo_minus_RhoDesign"] = {
        "best": paired(cry["PACoAptamer"][c]["recovery_best"]
                       - cry["RhoDesign"][c]["recovery_best"] for c in overlap),
        "t01": paired(cry["PACoAptamer"][c]["recovery_t01_mean"]
                      - cry["RhoDesign"][c]["recovery_t01_mean"] for c in overlap),
    }
    # 每方法 predbb−crystal（同链配对）
    pred_minus_cry = {}
    for m in METHODS:
        pred_minus_cry[m] = {
            "best": paired(predbb[m][c]["recovery_best"]
                           - cry[m][c]["recovery_best"] for c in overlap),
            "t01": paired(predbb[m][c]["recovery_t01_mean"]
                          - cry[m][c]["recovery_t01_mean"] for c in overlap),
        }
    # 优势分解：adv_predbb − adv_crystal（PACo−RhoDesign 优势在两种骨架下的差）
    adv_decomp = {
        "best": paired(
            (predbb["PACoAptamer"][c]["recovery_best"]
             - predbb["RhoDesign"][c]["recovery_best"])
            - (cry["PACoAptamer"][c]["recovery_best"]
               - cry["RhoDesign"][c]["recovery_best"]) for c in overlap),
        "t01": paired(
            (predbb["PACoAptamer"][c]["recovery_t01_mean"]
             - predbb["RhoDesign"][c]["recovery_t01_mean"])
            - (cry["PACoAptamer"][c]["recovery_t01_mean"]
               - cry["RhoDesign"][c]["recovery_t01_mean"]) for c in overlap),
    }
    cry_decomp = {
        "note": "同 32 链的晶体骨架口径（btest_v2_main_table per_chain）；"
                "predbb_minus_crystal 为逐方法骨架切换效应（配对）；"
                "advantage_decomposition = (PACo−RhoDesign)_predbb − "
                "(PACo−RhoDesign)_crystal，衡量优势变化中骨架的贡献",
        "crystal_summary": cry_sum,
        "predbb_minus_crystal": pred_minus_cry,
        "advantage_decomposition": adv_decomp,
    }
    print(f"[crystal zs] PACo best={cry_sum['methods']['PACoAptamer']['best']['mean']:.4f} "
          f"Rho best={cry_sum['methods']['RhoDesign']['best']['mean']:.4f} "
          f"Δbest={cry_sum['delta_PACo_minus_RhoDesign']['best']['mean']:+.4f}")
    print(f"[adv decomp] best {adv_decomp['best']['mean']:+.4f} "
          f"(p={adv_decomp['best']['wilcoxon_p_two_sided']:.4g}) "
          f"t01 {adv_decomp['t01']['mean']:+.4f} "
          f"(p={adv_decomp['t01']['wilcoxon_p_two_sided']:.4g})")

    # ---- 逐链明细（193 链全保留）
    per_chain = {}
    for c in sorted(base):
        rec = {"is_zs": c in zs, "len": predbb["PACoAptamer"][c].get("len"),
               "predbb": {}, "crystal": {}}
        for m in METHODS:
            rec["predbb"][m] = {
                "best": predbb[m][c]["recovery_best"],
                "t01": predbb[m][c]["recovery_t01_mean"]}
            rec["crystal"][m] = {
                "best": maintab[m][c]["recovery_best"],
                "t01": maintab[m][c]["recovery_t01_mean"]}
        rec["delta_predbb_PACo_minus_RhoDesign"] = {
            "best": rec["predbb"]["PACoAptamer"]["best"] - rec["predbb"]["RhoDesign"]["best"],
            "t01": rec["predbb"]["PACoAptamer"]["t01"] - rec["predbb"]["RhoDesign"]["t01"]}
        rec["delta_crystal_PACo_minus_RhoDesign"] = {
            "best": rec["crystal"]["PACoAptamer"]["best"] - rec["crystal"]["RhoDesign"]["best"],
            "t01": rec["crystal"]["PACoAptamer"]["t01"] - rec["crystal"]["RhoDesign"]["t01"]}
        rec["predbb_minus_crystal"] = {m: {
            "best": rec["predbb"][m]["best"] - rec["crystal"][m]["best"],
            "t01": rec["predbb"][m]["t01"] - rec["crystal"][m]["t01"]} for m in METHODS}
        per_chain[c] = rec

    out = {
        "meta": {
            "step": 105,
            "question": "预测骨架设定下，zero-shot 子集上 PACo 是否仍领先 RhoDesign？",
            "predbb_sources": PREDBB_FILES,
            "predbb_definition": "193 条 cluster-based test 链，晶体骨架换成 RhoFold+ "
                                 "对 native 序列的预测骨架后重跑五方法；逐链值取各文件 "
                                 "per_chain 的 recovery_best / recovery_t01_mean",
            "zs_definition": "95_lofo.json zs_table.*.vs_baseline.per_chain_delta 的 93 链 "
                             "（配体在训练中从未出现）；与 93_tanimoto_zs.json per_chain 键一致",
            "crystal_source": "btest_v2_main_table.json per_chain（477 链主表，晶体骨架）",
            "stats": "mean/median/bootstrap ci95（n=2000, seed=0，同 104i 惯例）；"
                     "配对 Δ 用 Wilcoxon 符号秩双侧（scipy 默认 zero_method=wilcox，"
                     "n<10 不做）",
            "date": "2026-10-16",
        },
        "sets": sets_block,
        "validation": validation,
        "zs_subset": zs_sum,
        "non_zs_subset": nz_sum,
        "full193": full_sum,
        "zs_subset_crystal_decomp": cry_decomp,
        "per_chain": per_chain,
    }
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, OUT)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
