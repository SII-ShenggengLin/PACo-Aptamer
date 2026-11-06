# -*- coding: utf-8 -*-
"""PACo-Aptamer 论文图表公共绘图规范(Nature Methods 落地)。

统一 12 色莫兰迪色板(全项目唯一取色来源):
  ["#B5838D", "#D3B5B0", "#C08B7C", "#D9C3A9", "#CDBF9E", "#A3B18A",
   "#8DA290", "#8FA6BB", "#6B7B8D", "#A99AAF", "#B7B0A2", "#7A7672"]
色板语义恒定:
- PACo 主色石板蓝 #6B7B8D(浅伴 #8FA6BB)
- 基线系列:RhoDesign 深灰 #7A7672(最强对手)、AlignIF 灰紫 #A99AAF、
  NA-MPNN 暖灰 #B7B0A2、gRNAde 沙 #CDBF9E
- 天然结构恒灰绿 #A3B18A(浅伴 #8DA290);配体/靶标恒陶土橙 #C08B7C;
  负对照恒灰玫红 #B5838D;辅助浅填充 #D3B5B0 / #D9C3A9
统计图规范:白底、无网格、细轴线、柱顶标数值、最优加粗、
小提琴+抖动点+黑色均值线、Wilcoxon P 面板内标注、Spearman+95%CI 角标;
图内文字一律黑色。
"""
import json
import os

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
OUT = os.path.join(ROOT, "figures", "out")

# ---------- 12 色板(全部用色的唯一来源) ----------
PALETTE = ["#B5838D", "#D3B5B0", "#C08B7C", "#D9C3A9", "#CDBF9E", "#A3B18A",
           "#8DA290", "#8FA6BB", "#6B7B8D", "#A99AAF", "#B7B0A2", "#7A7672"]

# ---------- 色板语义恒定 ----------
PACO = "#6B7B8D"        # 石板蓝主色
PACO_D = "#8FA6BB"      # 主色的浅伴(双口径柱/第二系列)
RHODESIGN = "#7A7672"   # 最深灰 = 最强对手
ALIGNIF = "#A99AAF"     # 灰紫
NAMPNN = "#B7B0A2"      # 暖灰
GRNADE = "#CDBF9E"      # 沙
NATIVE = "#A3B18A"      # 天然 = 灰绿
NEG = "#B5838D"         # 负对照/随机参照 = 灰玫红
GRAY = "#7A7672"        # 中性灰(温度深浅图例等非方法语义用色)
LIGAND = "#C08B7C"      # 配体/靶标 = 陶土橙
# 辅助(语义细分时取用)
AUX_CYAN = "#8FA6BB"; AUX_CORAL = "#B5838D"; AUX_ORANGE = "#C08B7C"
AUX_PERIW = "#A99AAF"; AUX_SAGE = "#8DA290"; AUX_SAND = "#D9C3A9"
DESIGN_BLUE = "#8FA6BB"  # 设计结构 = 浅石板蓝

METHOD_COLOR = {
    "PACo-Aptamer": PACO,
    "PACo": PACO,
    "RhoDesign": RHODESIGN,
    "AlignIF": ALIGNIF,
    "NA-MPNN": NAMPNN,
    "gRNAde": GRNADE,
}
METHOD_LABEL = {
    "PACo-Aptamer": "PACo-Aptamer",
    "PACo": "PACo-Aptamer",
    "RhoDesign": "RhoDesign",
    "AlignIF": "AlignIF",
    "NA-MPNN": "NA-MPNN",
    "gRNAde": "gRNAde",
}
# 展示顺序: PACo 在前, 基线按成绩降序
METHOD_ORDER = ["PACo-Aptamer", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]

# 莫兰迪蓝渐变(白色→浅伴→石板蓝):hexbin / 密度类图,端点取自色板
PACO_BLUES = LinearSegmentedColormap.from_list(
    "paco_blues", ["#FFFFFF", "#DCE4EC", "#B9C9D8", "#8FA6BB", "#6B7B8D"])
if "paco_blues" not in mpl.colormaps:
    mpl.colormaps.register(PACO_BLUES, name="paco_blues")

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
        "axes.titlesize": 6.0,
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


def panel_label(ax, letter, x=None, y=None, size=8):
    """Nature Methods 风格粗体小写 panel 字母。

    字母锚定在子图包围盒左上角外 1.2mm(以图像物理坐标计算):
    同一行子图 y1 相同 → 字母基线自动水平对齐;同一列 x0 相同 → 自动
    垂直对齐。x/y 仅为兼容旧调用保留,不再使用。
    """
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = fig.get_figwidth(), fig.get_figheight()
    dx = (1.2 * MM) / w_in
    dy = (1.2 * MM) / h_in
    fig.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=size,
             fontweight="bold", va="bottom", ha="left", color="black")


def load(rel):
    with open(os.path.join(SUR, rel)) as f:
        return json.load(f)


def save(fig, name, dpi=600, pad=0.02):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=pad)
    print("saved", name)


def pfmt(p):
    if p < 1e-4:
        return f"P = {p:.0e}".replace("e-", " × 10^−")
    if p < 0.001:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"
