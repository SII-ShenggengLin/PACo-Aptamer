# -*- coding: utf-8 -*-
"""第 108 步：预测骨架基准的骨架质量依赖（Spearman, per-chain recovery vs sctm_native）。

动机：94/100-103 号 JSON 中只有 94(PACo) 存了 spearman_vs_sctm；正文需要
五方法同口径对比。此处统一从 86_sctm.json(sctm_native) + 各方法 predbb JSON
+ 63_main_table_gfix.json(晶体主表,gRNAde 为 C/G 解码修正版)重算四个变体:
  sctm vs predbb_best / predbb_t01 / delta_best / delta_t01
(delta = predbb - crystal, 同链配对)。

校验:PACo 必须复现 94 号存储值 (-0.2135/+0.2156/+0.1749/+0.2074);
AlignIF best 必须复现 analysis/23 §4 的 0.694 (p=4.5e-29)。

产物:data/survey/108_backbone_quality_spearman.json(临时文件 + rename)。
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

SUR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey")
OUT = os.path.join(SUR, "108_backbone_quality_spearman.json")
TMP = OUT + ".tmp"

METHOD_FILES = {
    "PACo-Aptamer": "94_predbb_paco.json",
    "RhoDesign": "101_rhodesign_predbb.json",
    "gRNAde": "103_grnade_predbb.json",
    "NA-MPNN": "102_nampnn_predbb.json",
    "AlignIF": "100_alignif_predbb.json",
}
# 63_main_table_gfix 主表的 per_chain 方法键 -> 文件名方法名
MAIN_TAB_KEYS = {
    "PACo-Aptamer": "PACoAptamer",
    "RhoDesign": "RhoDesign",
    "gRNAde": "gRNAde",
    "NA-MPNN": "NA-MPNN",
    "AlignIF": "AlignIF",
}


def main():
    sctm = json.load(open(os.path.join(SUR, "86_sctm.json")))["per_chain"]
    mt = json.load(open(os.path.join(SUR, "63_main_table_gfix.json")))
    pc63 = mt["per_chain"]

    # 确认 63 主表逐链结构:per_chain[方法][链]
    print("63 per_chain 方法键:", list(pc63.keys()))

    result = {"task": "backbone-quality dependence (Spearman) on the 193-chain "
                      "predicted-backbone benchmark; delta = predbb - crystal "
                      "(crystal from 63_main_table_gfix, gRNAde C/G-decode-fixed)",
              "n_chains": None, "methods": {}}

    common_all = None
    for meth, fn in METHOD_FILES.items():
        pc = json.load(open(os.path.join(SUR, fn)))["per_chain"]
        crys = pc63[MAIN_TAB_KEYS[meth]]
        common = sorted(s for s in pc if s in sctm and s in crys)
        if common_all is None:
            common_all = set(common)
        else:
            common_all &= set(common)
        xs = np.array([sctm[s]["sctm_native"] for s in common])
        yb = np.array([pc[s]["recovery_best"] for s in common])
        yt = np.array([pc[s]["recovery_t01_mean"] for s in common])
        cb = np.array([crys[s]["recovery_best"] for s in common])
        ct = np.array([crys[s]["recovery_t01_mean"] for s in common])
        entry = {"n": len(common)}
        for name, y in [("sctm_vs_predbb_best", yb), ("sctm_vs_predbb_t01", yt),
                        ("sctm_vs_delta_best", yb - cb), ("sctm_vs_delta_t01", yt - ct)]:
            r = spearmanr(xs, y)
            entry[name] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        result["methods"][meth] = entry
        print(meth, len(common),
              {k: (round(v["rho"], 4), f"{v['p']:.1e}") for k, v in entry.items()
               if k != "n"})

    result["n_chains"] = len(common_all)
    result["chains"] = sorted(common_all)

    # 校验
    t = result["methods"]["PACo-Aptamer"]
    assert abs(t["sctm_vs_predbb_best"]["rho"] - (-0.21349171196918765)) < 1e-9, "PACo best 复现失败"
    assert abs(t["sctm_vs_predbb_t01"]["rho"] - 0.21563223648044522) < 1e-9, "PACo t01 复现失败"
    a = result["methods"]["AlignIF"]
    assert abs(a["sctm_vs_predbb_best"]["rho"] - 0.694) < 2e-3, "AlignIF best 复现失败"
    print("校验通过:PACo 四变体逐位复现 94 号存储值;AlignIF best=0.694 复现 analysis/23")

    with open(TMP, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(TMP, OUT)
    print("写出", OUT)


if __name__ == "__main__":
    main()
