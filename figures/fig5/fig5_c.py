# -*- coding: utf-8 -*-
"""Fig 5c | 新颖性双参照:每方法对训练库 vs AptaDB(已知 SELEX 适配体)
的最大序列身份中位数哑铃点图(实心 = 训练库,空心 = AptaDB)。

原 b 面板右缘的 AptaDB 中位数刻度移到这里,成为独立结果面板,
与正文"The distance to known SELEX aptamers in AptaDB is similar"对应;
纵轴上限加长留出顶部图例带,图例置于左上坐标范围内。
单独运行:python fig5_c.py → out/fig5_c.{pdf,png}(与拼合版物理尺寸一致)
"""
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import GRAY, METHOD_COLOR, METHOD_ORDER, panel_label, save


def medians(m):
    k = "PACo" if m == "PACo-Aptamer" else m
    e = C.NOV["methods"][k]
    return e["train"]["median"], e["aptadb"]["median"]


def draw(fig, gs):
    axc = fig.add_subplot(gs[0, 2])
    panel_label(axc, "c")
    ys = list(range(len(METHOD_ORDER) - 1, -1, -1))  # 首法在上
    for y, m in zip(ys, METHOD_ORDER):
        mt, ma = medians(m)
        c = METHOD_COLOR[m]
        axc.plot([mt, ma], [y, y], color=c, lw=0.7, alpha=0.55, zorder=2)
        axc.scatter([mt], [y], s=17, c=[c], zorder=3)
        axc.scatter([ma], [y], s=17, facecolors="white", edgecolors=c,
                    linewidths=0.8, zorder=4)
    axc.set_yticks(ys)
    axc.set_yticklabels(METHOD_ORDER)
    axc.set_ylim(-0.6, len(METHOD_ORDER) - 1 + 1.5)  # 顶部图例带
    axc.set_xlim(0.42, 0.56)
    axc.set_xticks([0.44, 0.48, 0.52, 0.56])
    axc.set_xlabel("Median maximum sequence identity")
    handles = [Line2D([], [], marker="o", ls="", markersize=3.4,
                      markerfacecolor=GRAY, markeredgecolor=GRAY,
                      label="training library"),
               Line2D([], [], marker="o", ls="", markersize=3.4,
                      markerfacecolor="white", markeredgecolor=GRAY,
                      markeredgewidth=0.8, label="AptaDB")]
    axc.legend(handles=handles, loc="upper left", fontsize=5.5,
               handlelength=1.0, labelspacing=0.3, handletextpad=0.4,
               borderaxespad=0.3)
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_c", pad=0.05)
