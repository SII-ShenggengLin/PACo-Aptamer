# -*- coding: utf-8 -*-
"""Extended Data 图公共模块:画布几何 + 数据加载 + 统计标注格式。

四张 ED 图(ed_fig1..ed_fig4)共用,结构对齐 figures/fig4/fig4_common.py:
- make_canvas(w_mm, h_mm, nrows, ncols, **gs_kw):统一 183mm 宽画布。
- 数据源一次性加载(data/survey/),各面板脚本直接用模块级常量。
- pfmt_sci / sig_star:P 值与显著性星号的统一格式(Δ 一律百分数)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load  # noqa: E402

# ---------- 数据 ----------
T93 = load("93_tanimoto_zs.json")            # ED1a/b 化学新颖性
R106 = load("106_tanimoto_robust.json")      # ED1c 指纹稳健性
LOFO = load("95_lofo.json")                  # ED2 LOFO + 覆盖曲线
FAM95 = load("95_families.json")             # ED2 家族定义
PRED = {                                     # ED3 五方法 predbb per_chain
    "PACo-Aptamer": load("94_predbb_paco.json"),
    "RhoDesign": load("101_rhodesign_predbb.json"),
    "gRNAde": load("103_grnade_predbb.json"),
    "AlignIF": load("100_alignif_predbb.json"),
    "NA-MPNN": load("102_nampnn_predbb.json"),
}
ZS105 = load("105_predbb_zs.json")           # ED3d zs 交叉切片(亦含晶体对照)
SCTM = load("86_sctm_gfix.json")             # ED3c 骨架质量(scTM)
MAIN63 = load("63_main_table_gfix.json")     # ED3a/b 晶体口径(gRNAde C/G 修正版)
SS97 = load("97_ss_robust.json")             # ED4a SS 扰动
ATT96 = load("96_attention.json")            # ED4b 配体注意力质量

PRED_ORDER = ["PACo-Aptamer", "RhoDesign", "gRNAde", "AlignIF", "NA-MPNN"]
PRED_KEY105 = {"PACo-Aptamer": "PACoAptamer", "RhoDesign": "RhoDesign",
               "gRNAde": "gRNAde", "AlignIF": "AlignIF", "NA-MPNN": "NA-MPNN"}
PRED_LAB = {"PACo-Aptamer": "PACo", "RhoDesign": "RhoDesign", "gRNAde": "gRNAde",
            "AlignIF": "AlignIF", "NA-MPNN": "NA-MPNN"}


def make_canvas(h_mm, nrows, ncols, **gs_kw):
    """183mm 全宽画布;h_mm 高度;返回 (fig, gs)。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(183 * MM, h_mm * MM))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gs_kw)
    return fig, gs


def pfmt_sci(p):
    """P 值格式:≥0.995 → P > 0.99;≥0.05 → P = 0.42;0.01–0.05 → P = 0.022;
    <0.01 → P = 5.4×10⁻⁵。"""
    if p is None:
        return "P = n.a."
    if p >= 0.995:
        return "P > 0.99"
    if p >= 0.05:
        return f"P = {p:.2f}"
    if p >= 0.01:
        return f"P = {p:.3f}"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10 ** exp
    return rf"$P = {mant:.1f}\times10^{{{exp}}}$"


def sig_star(p):
    if p is None:
        return "n.t."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def crystal_recovery(method):
    """63_main_table_gfix per_chain:{stem: recovery_best/t01_mean}(gRNAde 已修正)。"""
    return MAIN63["per_chain"][PRED_KEY105[method]]


def predbb_recovery(method):
    """五方法 predbb per_chain 统一为 {stem: (best, t01)}。"""
    pc = PRED[method]["per_chain"]
    return {k: (v["recovery_best"], v["recovery_t01_mean"]) for k, v in pc.items()}
