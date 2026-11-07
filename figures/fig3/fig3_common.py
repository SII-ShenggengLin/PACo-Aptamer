"""Fig 3 公共模块:统一画布几何 + 数据准备 + 三联面板共用绘图函数。"""
import json
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (
    DATA_DIR, MM, METHOD_COLOR, METHOD_ORDER, NEG, ROOT, PACO, apply_style, load,
    panel_label)

FIG_W_IN = 183 * MM
FIG_H_IN = 159 * MM

GRID_KW = dict(hspace=0.40, wspace=0.42,
               left=0.078, right=0.98, top=0.965, bottom=0.065)

def tighten_ylabels(fig):
    """列间呼吸感:y 刻度签 pad 3.5→1.5 pt、y 轴标签 labelpad 4→1.5 pt。"""
    for ax in fig.axes:
        ax.tick_params(axis="y", pad=1.5)
        lbl = ax.yaxis.label.get_text()
        if lbl:
            ax.set_ylabel(lbl, labelpad=1.5)

def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×159mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(3, 6, figure=fig, **GRID_KW)
    return fig, gs

al = load("benchmark_temporal_ligand.json")
pc = al["per_chain"]
wil = load("benchmark_statistics.json")
sp = json.load(open(os.path.join(DATA_DIR, "splits", "splits.json")))
views = al["views"]
MKEY = {m: ("PACoAptamer" if m == "PACo-Aptamer" else m) for m in METHOD_ORDER}

sys.path.insert(0, os.path.join(ROOT, "figures", "ed"))
import ed_common as ED

STEMS193 = ED.ZS105["full193"]["chains"]
PRED_ORDER = ED.PRED_ORDER

def view_arrays(view):
    """返回 (stems, x=RhoDesign%, y=PACo%),按 stem 排序的逐链恢复率。"""
    stems = sorted(sp[view])
    x = np.array([pc[s]["RhoDesign"]["recovery_best"] for s in stems]) * 100
    y = np.array([pc[s]["PACoAptamer"]["recovery_best"] for s in stems]) * 100
    return stems, x, y

def draw_bars(ax, view, letter):
    """双口径柱状(best 实色 / t01 半透明)+ 中性样块图例(与 Fig. 2a 一致)。"""
    panel_label(ax, letter)
    vb = [100 * views[view][MKEY[m]]["best"] for m in METHOD_ORDER]
    vt = [100 * views[view][MKEY[m]]["t01"] for m in METHOD_ORDER]
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    ax.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           ec="white", lw=0.3)
    ax.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           alpha=0.45, ec="white", lw=0.3)
    for x, v in list(zip(xs - w / 2, vb)) + list(zip(xs + w / 2, vt)):
        ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=4.8)
    ax.legend(handles=[
        Patch(fc="0.25", ec="none", label="best sample per chain (solid)"),
        Patch(fc="0.25", alpha=0.45, ec="none",
              label="t = 0.1 mean (translucent)")],
        loc="upper right", fontsize=5.5, handlelength=1.4,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="center", va="top")
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.set_ylabel("Sequence recovery (%)")
    ax.set_ylim(0, 110)
    tighten_ylabels(ax.figure)

def draw_h2h(ax, view, letter):
    """head-to-head 逐链散点(PACo-Aptamer vs RhoDesign)。"""
    panel_label(ax, letter)
    _, x, y = view_arrays(view)
    lim = (40, 100)
    ax.plot(lim, lim, color="#7A7672", lw=0.6, ls="--", zorder=1)
    ax.scatter(x[y > x], y[y > x], s=3.5, c=PACO, alpha=0.55, lw=0, zorder=2)
    ax.scatter(x[y <= x], y[y <= x], s=3.5, c="#B7B0A2", alpha=0.55, lw=0,
               zorder=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=PACO,
                              markeredgecolor="none", markersize=3,
                              alpha=0.8, label="PACo-Aptamer higher"),
                       Line2D([], [], color="#7A7672", lw=0.6, ls="--",
                              label="equal recovery")],
              loc="lower right", fontsize=5.5, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RhoDesign recovery (%)")
    ax.set_ylabel("PACo-Aptamer recovery (%)")
    tighten_ylabels(ax.figure)

def draw_hist(ax, view, letter):
    """逐链恢复率优势(PACo-Aptamer − RhoDesign)直方图,正负分色(与 Fig. 2f 同款)。"""
    panel_label(ax, letter)
    _, x, y = view_arrays(view)
    dv = (y - x)
    edges = np.arange(-20, 20.01, 2.5)
    centers = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(dv, bins=edges)
    share = 100 * counts / len(dv)
    tie_share = 100 * np.mean(dv == 0)

    share[centers == 1.25] -= tie_share
    pos = centers > 0
    ax.bar(centers[pos], share[pos], width=2.3, color=PACO, ec="white",
           lw=0.3)
    ax.bar(centers[~pos], share[~pos], width=2.3, color=NEG, ec="white",
           lw=0.3)
    ax.bar([0], [tie_share], width=0.9, color="#B7B0A2", ec="white", lw=0.3)
    ax.axvline(0, color="#7A7672", lw=0.6, ls=":")
    ax.axvline(np.mean(dv), color="black", lw=0.9)
    ax.legend(handles=[
        Patch(fc=PACO, ec="none", label="PACo-Aptamer higher"),
        Patch(fc=NEG, ec="none", label="RhoDesign higher"),
        Patch(fc="#B7B0A2", ec="none", label="equal recovery")],
        loc="upper left", fontsize=5.5, handlelength=1.2,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 30)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Advantage over RhoDesign (%)")
    ax.set_ylabel("Chains (%)")
    tighten_ylabels(ax.figure)
