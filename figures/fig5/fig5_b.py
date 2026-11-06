# -*- coding: utf-8 -*-
"""Fig 5b | 新颖性:rank-ordered 最大训练库相似度曲线。

图例只放方法名(统计量由图题承载);identity = 0.8 参照线入图例,
不再贴线标注;AptaDB 对比移至专门的 c 面板。
单独运行:python fig5_b.py → out/fig5_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, NEG, panel_label, save


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 1])
    panel_label(axb, "b")
    for m in METHOD_ORDER:
        k = "PACo" if m == "PACo-Aptamer" else m
        rc = np.array(C.NOV["methods"][k]["train"]["rank_curve"])
        xq = np.linspace(0, 1, len(rc))
        axb.plot(xq, rc, lw=0.9, color=METHOD_COLOR[m], label=m)
    axb.axhline(0.8, color=NEG, lw=0.6, ls=":", zorder=1)
    handles = [Line2D([], [], color=METHOD_COLOR[m], lw=0.9, label=m)
               for m in METHOD_ORDER]
    handles.append(Line2D([], [], ls=":", color=NEG, lw=0.6,
                          label="identity = 0.8"))
    axb.legend(handles=handles, loc="upper left", fontsize=5.5, ncol=2,
               handlelength=1.5, labelspacing=0.3, handletextpad=0.5,
               columnspacing=0.9, borderaxespad=0.3)
    axb.set_xlim(0, 1); axb.set_ylim(0, 1.32)  # 上方留白带给图例
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axb.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb.set_xlabel("Designs ranked by maximum identity\nto the training library")
    axb.set_ylabel("Maximum sequence identity")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_b", pad=0.05)
