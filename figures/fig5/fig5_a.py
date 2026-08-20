# -*- coding: utf-8 -*-
"""Fig 5a | 温度扫描 recovery–diversity 前沿(TACo / RhoDesign / gRNAde)。

温度渐变着色标记;标注避让数据点( uniq 0.88 属 TACo t0.1、RhoDesign t0.1
uniq 0.30 → 70% 重复 = 模式坍缩,箭头分别指向各自数据点)。
单独运行:python fig5_a.py → out/fig5_a.{pdf,png}(与拼合版物理尺寸一致)
"""
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import (METHOD_COLOR, RHODESIGN, TACO, panel_label, save)

TEMPS = ["t=0.1", "t=0.5", "t=1.0"]


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0:2])
    panel_label(axa, "a", x=-0.19, y=1.12)
    for m in ["TACo", "RhoDesign", "gRNAde"]:
        xs = [100 * C.TD[m][t]["div"]["mean"] for t in TEMPS]
        ys = [100 * C.TD[m][t]["rec"]["mean"] for t in TEMPS]
        base = METHOD_COLOR[m]
        shades = [base, C.lighten(base, 0.35), C.lighten(base, 0.62)]
        axa.plot(xs, ys, color=base, lw=0.7, alpha=0.7, zorder=2)
        for x, y, c, t in zip(xs, ys, shades, TEMPS):
            axa.scatter(x, y, s=16, c=[c], edgecolors=base, linewidths=0.6,
                        zorder=3)
    # TACo t0.1 点 (23.6, 59.7):文字置于其上方空白区,白底衬保证可读
    axa.annotate("t = 0.1, uniq 0.88", (23.6, 59.7), fontsize=5.0, color=TACO,
                 xytext=(15.5, 65.2), textcoords="data", ha="center",
                 bbox=dict(fc="white", ec="none", pad=1.0),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=TACO,
                                 shrinkA=2, shrinkB=4))
    axa.annotate("t = 1.0", (31.4, 58.5), fontsize=5.0, color=TACO,
                 xytext=(33.6, 63.4), textcoords="data", ha="right",
                 bbox=dict(fc="white", ec="none", pad=1.0),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=TACO,
                                 shrinkA=2, shrinkB=4))
    # RhoDesign t0.1 (8.2, 54.6) uniq 0.30 → 模式坍缩;文字放中部空白区
    # (gRNAde t0.1 点在 (16.1, 31.5),避开其上下方)
    axa.annotate("mode collapse at t = 0.1:\n70% duplicate sequences",
                 (8.2, 54.6), fontsize=4.8, color=RHODESIGN,
                 xytext=(27.5, 41.5), textcoords="data", ha="center",
                 bbox=dict(fc="white", ec="none", pad=1.0),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=RHODESIGN,
                                 ls=":", connectionstyle="arc3,rad=-0.18",
                                 shrinkA=2, shrinkB=4))
    axa.axhline(25, color="#B07060", lw=0.5, ls=":")
    axa.text(5.0, 26.0, "random", fontsize=4.6, color="#B07060", ha="left")
    axa.set_xlabel("Pool diversity (mean pairwise distance)")
    axa.set_ylabel("Recovery, t-mean (%)")
    axa.set_xlim(4, 36)
    axa.set_ylim(22, 69)
    handles = [Line2D([], [], marker="o", ls="-", color=METHOD_COLOR[m],
                      label=m if m != "TACo" else "TACo-Aptamer",
                      markersize=3.5,
                      markerfacecolor=C.lighten(METHOD_COLOR[m], 0.35))
               for m in ["TACo", "RhoDesign", "gRNAde"]]
    axa.legend(handles=handles, loc="lower left", fontsize=5.0)
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_a")
