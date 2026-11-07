"""Fig 5d | 生成池 4-mer 组成 t-SNE(60 链 × 5 方法 × 30 设计 + native ×)。"""
import numpy as np
from matplotlib.lines import Line2D

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, NATIVE, panel_label, save

def draw(fig, gs):
    axd = fig.add_subplot(gs[1, 0])
    panel_label(axd, "d")
    lab = np.array(C.TS["label"]); x = np.array(C.TS["x"]); y = np.array(C.TS["y"])
    for m in METHOD_ORDER:
        key = "PACo" if m == "PACo-Aptamer" else m
        sel = lab == key
        axd.scatter(x[sel], y[sel], s=2.0, c=METHOD_COLOR[m], alpha=0.35, lw=0)
    sel = lab == "native"
    axd.scatter(x[sel], y[sel], s=16, marker="x", c=NATIVE, lw=0.8, zorder=5)
    axd.set_xticks([]); axd.set_yticks([])
    axd.set_xlabel("t-SNE 1"); axd.set_ylabel("t-SNE 2")
    axd.set_xlim(-112, 113)
    axd.set_ylim(-116, 124)
    handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLOR[m],
                      markersize=3, label=m) for m in METHOD_ORDER]
    handles.append(Line2D([], [], marker="x", ls="", color=NATIVE,
                          markersize=4, markeredgewidth=1.0, label="native"))
    axd.legend(handles=handles, ncol=3, loc="upper center", fontsize=5.5,
               handlelength=0.9, handletextpad=0.25, columnspacing=0.8,
               labelspacing=0.2, borderaxespad=0.3)
    return axd

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_d", pad=0.05)
