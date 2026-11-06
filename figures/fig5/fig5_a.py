# -*- coding: utf-8 -*-
"""Fig 5a | 温度扫描 recovery–diversity 前沿(PACo-Aptamer / RhoDesign / gRNAde)。

图内不放数字标注(数字由图题承载):温度由标记深浅编码(深→浅 =
t = 0.1→1.0),随机参照线入图例(仅标 random);图例置于左上,
纵轴上限加长留出图例带;横轴取原始归一化编辑距离,
与正文引用的 0.236/0.082 等数值同口径。
单独运行:python fig5_a.py → out/fig5_a.{pdf,png}(与拼合版物理尺寸一致)
"""
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import (GRAY, METHOD_COLOR, NEG, panel_label, save)

TEMPS = ["t=0.1", "t=0.5", "t=1.0"]


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0])
    panel_label(axa, "a")
    for m in ["PACo-Aptamer", "RhoDesign", "gRNAde"]:
        k = "PACo" if m == "PACo-Aptamer" else m
        xs = [C.TD[k][t]["div"]["mean"] for t in TEMPS]
        ys = [100 * C.TD[k][t]["rec"]["mean"] for t in TEMPS]
        base = METHOD_COLOR[m]
        shades = [base, C.lighten(base, 0.35), C.lighten(base, 0.62)]
        axa.plot(xs, ys, color=base, lw=0.7, alpha=0.7, zorder=2)
        for x, y, c in zip(xs, ys, shades):
            axa.scatter(x, y, s=16, c=[c], edgecolors=base, linewidths=0.6,
                        zorder=3)
    axa.axhline(25, color=NEG, lw=0.6, ls=":", zorder=1)
    # 图例单列置于左上;纵轴上限加长出顶部图例带(数据点最高 y≈60)
    temp_tuple = tuple(
        Line2D([], [], marker="o", ls="", markersize=3.2,
               markerfacecolor=s, markeredgecolor=GRAY, markeredgewidth=0.4)
        for s in (GRAY, C.lighten(GRAY, 0.35), C.lighten(GRAY, 0.62)))
    handles = [
        Line2D([], [], marker="o", ls="-", color=METHOD_COLOR["PACo-Aptamer"],
               markersize=3.2, markerfacecolor=C.lighten(
                   METHOD_COLOR["PACo-Aptamer"], 0.35)),
        Line2D([], [], marker="o", ls="-", color=METHOD_COLOR["RhoDesign"],
               markersize=3.2, markerfacecolor=C.lighten(
                   METHOD_COLOR["RhoDesign"], 0.35)),
        Line2D([], [], marker="o", ls="-", color=METHOD_COLOR["gRNAde"],
               markersize=3.2, markerfacecolor=C.lighten(
                   METHOD_COLOR["gRNAde"], 0.35)),
        temp_tuple,
        Line2D([], [], ls=":", color=NEG, lw=0.6),
    ]
    labels = ["PACo-Aptamer", "RhoDesign", "gRNAde",
              "t = 0.1 → 1.0", "random"]
    axa.legend(handles=handles, labels=labels,
               handler_map={tuple: HandlerTuple()},
               loc="upper left", fontsize=5.5,
               handlelength=1.5, labelspacing=0.35,
               handletextpad=0.5, borderaxespad=0.3)
    axa.set_xlabel("Pool diversity\n(mean normalised edit distance)")
    axa.set_ylabel("Mean recovery (%)")
    axa.set_xlim(0.02, 0.72)
    axa.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    axa.set_ylim(22, 86)
    axa.set_yticks([30, 40, 50, 60, 70])
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_a", pad=0.05)
