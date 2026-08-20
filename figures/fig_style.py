# -*- coding: utf-8 -*-
"""TACo-Aptamer 论文六图公共绘图规范(蓝图 §六 莫兰迪落地)。

色板语义恒定:
- TACo 主色雾霾蓝 #4A7BA6 (ens2 深一档 #2E5F8A)
- 基线灰阶系列,RhoDesign 用最深灰作"最强对手"
- 配体/靶标恒橙;天然结构恒灰绿;设计结构恒雾蓝;负对照恒砖红
统计图规范:白底、无网格、细轴线、柱顶标数值、最优加粗、
小提琴+抖动点+黑色均值线、Wilcoxon P 面板内标注、Spearman+95%CI 角标。
"""
import json
import os

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
OUT = os.path.join(ROOT, "figures", "out")

# ---------- 色板(语义恒定) ----------
TACO = "#4A7BA6"        # 雾霾蓝主色
TACO_D = "#2E5F8A"      # ens2 深一档
RHODESIGN = "#545450"   # 最深灰 = 最强对手
ALIGNIF = "#8C8C88"
NAMPNN = "#A8A8A4"
GRNADE = "#C4C4C0"
NATIVE = "#7FA88B"      # 天然 = 灰绿
NEG = "#B07060"         # 负对照 = 砖红
LIGAND = "#E8A088"      # 配体/靶标 = 柔橙
# 辅助莫兰迪
AUX_CYAN = "#90C8D8"; AUX_CORAL = "#E88880"; AUX_ORANGE = "#E8A088"
AUX_PERIW = "#9888C8"; AUX_SAGE = "#A8C0A8"; AUX_SAND = "#D8C8B0"
DESIGN_BLUE = "#6C9BD1"  # 设计结构 = 雾蓝

METHOD_COLOR = {
    "TACo-Aptamer": TACO,
    "TACo": TACO,
    "RhoDesign": RHODESIGN,
    "AlignIF": ALIGNIF,
    "NA-MPNN": NAMPNN,
    "gRNAde": GRNADE,
}
METHOD_LABEL = {
    "TACo-Aptamer": "TACo-Aptamer",
    "TACo": "TACo-Aptamer",
    "RhoDesign": "RhoDesign",
    "AlignIF": "AlignIF",
    "NA-MPNN": "NA-MPNN",
    "gRNAde": "gRNAde",
}
# 展示顺序: TACo 在前, 基线按成绩降序
METHOD_ORDER = ["TACo-Aptamer", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]

# 莫兰迪蓝色渐变(白色→雾霾蓝):替换 matplotlib 自带 "Blues"(顶端高饱和
# 藏青 #08306b,非莫兰迪),用于 hexbin / 密度类图
TACO_BLUES = LinearSegmentedColormap.from_list(
    "taco_blues", ["#FFFFFF", "#E9EFF5", "#C6D6E4", "#9BB8CF", "#6E97B6",
                   "#4A7BA6", "#35607F"])
mpl.colormaps.register(TACO_BLUES, name="taco_blues")

# ---------- 全局 rc ----------
def apply_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        # Nature Methods 要求全文(含数学式)无衬线 Arial;mathtext 也走 Arial
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "mathtext.cal": "Arial:italic",
        "mathtext.default": "regular",
        "font.size": 6.0,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.2,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "legend.frameon": False,
        "lines.linewidth": 0.8,
        "pdf.fonttype": 42,
    })

MM = 1 / 25.4  # mm→inch;Nature Methods 双栏 183mm


def panel_label(ax, letter, x=-0.14, y=1.06, size=8):
    """Nature Methods 风格粗体小写 panel 字母。"""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")


def load(rel):
    with open(os.path.join(SUR, rel)) as f:
        return json.load(f)


def save(fig, name, dpi=350):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)
    print("saved", name)


def pfmt(p):
    if p < 1e-4:
        return f"P = {p:.0e}".replace("e-", " × 10^−")
    if p < 0.001:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"
