# -*- coding: utf-8 -*-
"""Fig 5c | 生成池 4-mer 组成 t-SNE(60 链 × 5 方法 × 30 设计 + native ×)。

单独运行:python fig5_c.py → out/fig5_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:3])
    panel_label(axc, "c", x=-0.13, y=1.12)
    lab = np.array(C.TS["label"]); x = np.array(C.TS["x"]); y = np.array(C.TS["y"])
    for m in METHOD_ORDER:
        key = "TACo" if m == "TACo-Aptamer" else m
        sel = lab == key
        axc.scatter(x[sel], y[sel], s=1.2, c=METHOD_COLOR[m], alpha=0.30, lw=0)
    sel = lab == "native"
    axc.scatter(x[sel], y[sel], s=14, marker="x", c="#3D5A45", lw=0.9,
                label="native", zorder=5)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_xlabel("t-SNE 1"); axc.set_ylabel("t-SNE 2")
    handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLOR[m],
                      markersize=3, alpha=0.9,
                      label=m if m != "TACo-Aptamer" else "TACo-Aptamer")
               for m in METHOD_ORDER]
    handles.append(Line2D([], [], marker="x", ls="", color="#3D5A45",
                          markersize=4, label="native"))
    axc.legend(handles=handles, ncol=3, loc="upper right", fontsize=4.8,
               handletextpad=0.2, columnspacing=0.7)
    axc.text(0.015, 0.02,
             "60 test chains × 30 designs per method (t = 0.1 pools), 4-mer composition",
             transform=axc.transAxes, fontsize=4.6, color="#545450")
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_c")
