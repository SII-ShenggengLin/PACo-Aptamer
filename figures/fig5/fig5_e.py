# -*- coding: utf-8 -*-
"""Fig 5e | 家族组成保真 KL(LOO TM 簇 profile)横向柱图。

横排方法名(与正文同名、不旋转);数值标签过 95% CI 上界后放置;
纵轴上限加长留出顶部图例带,random-sequence 参照虚线入图例(置于左上图例带),
虚线只画到柱区高度,不进图例带。
单独运行:python fig5_e.py → out/fig5_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, NEG, panel_label, save

Y_TOP = -0.6            # 柱区下界
Y_LEGEND_TOP = 5.15     # 加长后的纵轴上界(顶部图例带)


def draw(fig, gs):
    axe = fig.add_subplot(gs[1, 1])
    panel_label(axe, "e")
    order = sorted(METHOD_ORDER, key=lambda m: C.KL["methods"][
        "PACo" if m == "PACo-Aptamer" else m]["kl_mean"])
    ys = list(range(len(order) - 1, -1, -1))  # KL 最小(最优)在最上
    for y, m in zip(ys, order):
        e = C.KL["methods"]["PACo" if m == "PACo-Aptamer" else m]
        v, lo, hi = e["kl_mean"], e["ci95"][0], e["ci95"][1]
        c = METHOD_COLOR[m]
        axe.barh(y, v, 0.62, color=c, zorder=2)
        axe.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=c,
                     elinewidth=0.5, capsize=1.3, capthick=0.5, zorder=3)
        axe.text(hi + 0.09, y, f"{v:.2f}", va="center", ha="left",
                 fontsize=5.2, color="black")
    axe.set_yticks(ys)
    axe.set_yticklabels(order)
    axe.set_ylim(Y_TOP, Y_LEGEND_TOP)
    rr = C.KL["random_ref"]
    span = Y_LEGEND_TOP - Y_TOP
    axe.axvline(rr["kl_mean"], color=NEG, lw=0.7, ls="--", zorder=1,
                ymax=(len(order) - 1 + 0.45 - Y_TOP) / span)  # 只画到柱区
    axe.set_xlim(0, 5.3)
    axe.set_xticks([0, 1, 2, 3, 4, 5])
    axe.set_xlabel("KL divergence from the family profile")
    axe.legend([Line2D([], [], ls="--", color=NEG, lw=0.7)],
               ["random-sequence reference"], loc="upper left", fontsize=5.5,
               handlelength=1.6, handletextpad=0.5, borderaxespad=0.3)
    return axe


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_e", pad=0.05)
