# -*- coding: utf-8 -*-
"""Fig 1 公共模块:统一画布几何 + 共享绘图元件 + 数据。

所有子图脚本(fig1_a..e)与组合脚本(fig1_assemble)共用:
- make_canvas():同一张 183×138 mm 画布、同一 GridSpec(2,8),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- box/arrow/rna_icon/lig_icon:示意图元件(莫兰迪底色 + 细边框)。
- MAIN:btest_v2 主表(panel e 预览柱状)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import LIGAND, MM, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 138 * MM
GRID_KW = dict(hspace=0.42, wspace=0.9, height_ratios=[1.0, 1.12])


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×138mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 8, figure=fig, **GRID_KW)
    return fig, gs


# ---------- 示意图元件底色(莫兰迪) ----------
C_DATA, C_MODEL, C_MET, C_SS = "#F0EDE8", "#DCE6EF", "#E9E2D8", "#E4E9E1"
EC = "#8C8C88"


def box(ax, x, y, w, h, fc, txt, ec=EC, lw=0.5, fs=5.0, bold=False,
        style="round,pad=0.10", tc=None):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc or "black", fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#545450", lw=0.7, rad=0.0, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=8, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def rna_icon(ax, cx, cy, s=1.0, color="#545450", lw=0.9):
    """简化 RNA 折叠 icon(平滑路径)。"""
    t = np.linspace(0, 4 * np.pi, 120)
    x = cx + s * 0.09 * t * np.cos(t * 0.22)
    y = cy + s * 0.32 * np.sin(t)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round")


def lig_icon(ax, cx, cy, s=1.0, color=LIGAND):
    """小分子 icon(六元环 + 取代基)。"""
    ang = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    ax.plot(cx + s * 0.6 * np.cos(ang), cy + s * 0.6 * np.sin(ang),
            color=color, lw=1.0)
    for a in [0.3, 2.4, 4.5]:
        ax.plot([cx + s * 0.6 * np.cos(a), cx + s * 1.1 * np.cos(a)],
                [cy + s * 0.6 * np.sin(a), cy + s * 1.1 * np.sin(a)],
                color=color, lw=0.8)


# ---------- 数据 ----------
MAIN = load("btest_v2_main_table.json")["table"]
