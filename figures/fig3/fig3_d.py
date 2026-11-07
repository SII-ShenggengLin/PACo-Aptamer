"""Fig 3d | independent test 235 链:五方法双口径均值点图(Cleveland dot plot)。"""
import numpy as np
from matplotlib.lines import Line2D

import fig3_common as C
from fig_style import (METHOD_COLOR, METHOD_ORDER, panel_label, save)

def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 0:2])
    panel_label(ax, "d")
    stems = sorted(C.sp["independent_test"])
    for i, m in enumerate(METHOD_ORDER):
        mk = C.MKEY[m]
        col = METHOD_COLOR[m]
        for off, field in ((-0.16, "recovery_best"), (0.16, "recovery_t01_mean")):
            solid = field == "recovery_best"
            mean = 100 * np.mean([C.pc[s][mk][field] for s in stems])
            ax.plot(i + off, mean, marker="o", ms=4.4, mfc=col, mec="white",
                    mew=0.4, alpha=1.0 if solid else 0.45, ls="none",
                    zorder=3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.4, markerfacecolor="0.25",
               markeredgecolor="none",
               label="best sample per chain (filled)"),
        Line2D([], [], marker="o", ls="none", ms=4.4, markerfacecolor="0.25",
               markeredgecolor="none", alpha=0.45,
               label="t = 0.1 mean (translucent)")],
        loc="upper right", fontsize=5.5, handlelength=1.2,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="center", va="top")
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.set_ylabel("Sequence recovery (%)")
    ax.set_ylim(0, 110)
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_d")
