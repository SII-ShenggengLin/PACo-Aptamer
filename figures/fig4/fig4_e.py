# -*- coding: utf-8 -*-
"""Fig 4e | SS 条件保真 INF 柱状:native 上界 1.0 与 random 下界 0.03 改为
点状参照线(与 Fig. 2a 随机参照虚线同款),柱体只画五方法。

2026-10-22 起原 d 的双子图拆开各自成面板(e 多裁判柱状 / f 同一性分箱曲线),
宽度大幅增加;图内上下界角标删除(数值由柱旁标注与图注承载);
横坐标全名 PACo-Aptamer,居中 20°;数值标注不加粗。
单独运行:python fig4_e.py → out/fig4_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig4_common as C
from fig_style import METHOD_COLOR, NATIVE, NEG, panel_label, save

ROWS = [("gRNAde", "gRNAde"), ("NA-MPNN", "NA-MPNN"), ("AlignIF", "AlignIF"),
        ("PACo-Aptamer", "PACo_ens2"), ("RhoDesign", "RhoDesign")]


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "e")
    names, vals, cols = [], [], []
    cis_lo, cis_hi = [], []
    for lab, key in ROWS:
        d = C.P03[key]["inf"]
        names.append(lab); vals.append(d["mean"])
        cis_lo.append(d["mean"] - d["ci95"][0])
        cis_hi.append(d["ci95"][1] - d["mean"])
        cols.append(METHOD_COLOR[lab])
    x = np.arange(len(names))
    ax.bar(x, vals, 0.62, yerr=[cis_lo, cis_hi], color=cols,
           error_kw=dict(lw=0.5, capsize=1.4), ec="white", lw=0.3)
    for x0, v, e in zip(x, vals, cis_hi):
        ax.text(x0, v + e + 0.09, f"{v:.3f}", ha="center", fontsize=5.2)
    ax.axhline(C.P03["native"]["inf"]["mean"], color=NATIVE, lw=0.6, ls=":")
    ax.axhline(C.P03["random"]["inf"]["mean"], color=NEG, lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="center", va="top")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel("SS fidelity, INF")
    # 上限 1.25:图例(底≈1.10)整体位于 native 参考线(1.0)之上,不再被穿过
    ax.set_ylim(0, 1.25)
    ax.legend(handles=[
        Line2D([], [], color=NATIVE, lw=0.6, ls=":", label="native"),
        Line2D([], [], color=NEG, lw=0.6, ls=":", label="random")],
        loc="upper right", fontsize=6.0, handlelength=1.4,
        handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35)
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_e")
