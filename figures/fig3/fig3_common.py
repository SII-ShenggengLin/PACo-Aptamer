# -*- coding: utf-8 -*-
"""Fig 3 公共模块:统一画布几何 + 数据准备 + 三联面板共用绘图函数。

所有子图脚本(fig3_a..g)与组合脚本(fig3_assemble)共用:
- make_canvas():同一张 183×122 mm 画布、同一 GridSpec(2,10),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:alayer_v2_retrained(per_chain + views)、81_wilcoxon、splits_v2。
- draw_bars / draw_h2h / draw_violin:b–d 与 e–g 两组三联共用
  (zero_shot_targets 与 independent_test 各一套,布局完全相同)。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (  # noqa: E402
    MM, METHOD_COLOR, METHOD_ORDER, ROOT, TACO, apply_style, load,
    panel_label)

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 122 * MM
GRID_KW = dict(hspace=0.52, wspace=1.9)


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×122mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 10, figure=fig, **GRID_KW)
    return fig, gs


def pfmt_p(p):
    if p < 1e-4:
        m = int(np.floor(np.log10(p)))
        return f"P = {p/10**m:.1f} × 10$^{{{m}}}$"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


# ================= 数据 =================
al = load("alayer_v2_retrained.json")
pc = al["per_chain"]
wil = load("81_wilcoxon.json")
sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
views = al["views"]
MKEY = {m: ("TACoAptamer" if m == "TACo-Aptamer" else m) for m in METHOD_ORDER}


def view_arrays(view):
    """返回 (stems, x=RhoDesign%, y=TACo%),按 stem 排序的逐链恢复率。"""
    stems = sorted(sp[view])
    x = np.array([pc[s]["RhoDesign"]["recovery_best"] for s in stems]) * 100
    y = np.array([pc[s]["TACoAptamer"]["recovery_best"] for s in stems]) * 100
    return stems, x, y


# ================= 三联面板共用绘图 =================
def draw_bars(ax, view, wilkey, letter, title):
    """双口径柱状(best 实色 / t01 半透明)+ vs RhoDesign Wilcoxon 白底框。"""
    panel_label(ax, letter, x=-0.145, y=1.14)
    vb = [100 * views[view][MKEY[m]]["best"] for m in METHOD_ORDER]
    vt = [100 * views[view][MKEY[m]]["t01"] for m in METHOD_ORDER]
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    ax.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           ec="white", lw=0.3)
    ax.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           alpha=0.45, ec="white", lw=0.3)
    for x, v in zip(xs - w / 2, vb):
        ax.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8,
                fontweight="bold" if v == max(vb) else "normal")
    for x, v in zip(xs + w / 2, vt):
        ax.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8, alpha=0.75)
    wj = wil[wilkey]["RhoDesign"]["recovery_best"]
    wjt = wil[wilkey]["RhoDesign"]["recovery_t01_mean"]
    ax.text(0.98, 0.985,
            (f"vs RhoDesign: Δ = +{100*wj['delta']:.1f} pp, "
             f"{pfmt_p(wj['p_two_sided'])}\n(t = 0.1: Δ = +{100*wjt['delta']:.1f} pp, "
             f"{pfmt_p(wjt['p_two_sided'])})"),
            transform=ax.transAxes, fontsize=4.7, ha="right", va="top",
            bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("-Aptamer", "") for m in METHOD_ORDER],
                       rotation=30, ha="right")
    ax.set_ylabel("Sequence recovery (%)")
    ax.set_title(title, fontsize=5.6, pad=3.0)
    ax.set_ylim(0, 116)


def draw_h2h(ax, view, letter):
    """head-to-head 散点(TACo vs RhoDesign)。"""
    panel_label(ax, letter, x=-0.21, y=1.14)
    _, x, y = view_arrays(view)
    lim = (20, 100)
    ax.fill_between([0, 100], [0, 100], [100, 100], color="#EDF2F7", zorder=0)
    ax.plot(lim, lim, color="#8C8C88", lw=0.6, ls="--", zorder=1)
    ax.scatter(x[y <= x], y[y <= x], s=4, c=TACO, alpha=0.6, lw=0, zorder=2)
    ax.scatter(x[y > x], y[y > x], s=4, c="#B8B8B4", alpha=0.6, lw=0, zorder=2)
    win = float(np.mean(y > x))
    ax.text(0.06, 0.94, f"TACo-Aptamer wins {100*win:.0f}%\nof chains",
            transform=ax.transAxes, va="top", fontsize=5.2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("RhoDesign recovery (%)")
    ax.set_ylabel("TACo-Aptamer recovery (%)")


def draw_violin(ax, view, letter, rng):
    """逐链 Δ(ey − ex)小提琴 + 抖动点 + 黑色均值线。"""
    panel_label(ax, letter, x=-0.21, y=1.14)
    _, x, y = view_arrays(view)
    dv = (y - x)
    vp = ax.violinplot([dv], widths=0.75, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor("#DCE6EF"); b.set_edgecolor(TACO); b.set_alpha(0.9)
    ax.scatter(rng.normal(0, 0.05, len(dv)), dv, s=2.5, c=TACO, alpha=0.45,
               lw=0)
    ax.axhline(0, color="#545450", lw=0.6)
    ax.axhline(np.mean(dv), color="black", lw=0.9)
    ax.text(0.97, 0.955, f"mean Δ = {np.mean(dv):+.1f} pp\nmedian {np.median(dv):+.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0,
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xticks([]); ax.set_xlim(-0.65, 0.65)
    ax.set_ylabel("Δ recovery vs RhoDesign (pp)")
