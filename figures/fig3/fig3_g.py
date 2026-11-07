"""Fig 3g | 预测骨架 193 链:晶体 ↔ 预测配对哑铃图(五方法 × 双口径)。"""
import numpy as np
from matplotlib.lines import Line2D

import fig3_common as C
from fig_style import METHOD_COLOR, panel_label, save

def _predbb_means(method):
    m = C.ED.ZS105["full193"]["methods"][C.ED.PRED_KEY105[method]]
    return 100 * m["best"]["mean"], 100 * m["t01"]["mean"]

def _crystal_means(method):
    pc = C.ED.crystal_recovery(method)
    b = [pc[s]["recovery_best"] for s in C.STEMS193]
    t = [pc[s]["recovery_t01_mean"] for s in C.STEMS193]
    return 100 * np.mean(b), 100 * np.mean(t)

def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 0:2])
    panel_label(ax, "g")
    ms = C.PRED_ORDER
    for gi, m in enumerate(ms):
        pb, pt = _predbb_means(m)
        cb, ct = _crystal_means(m)
        col = METHOD_COLOR[m]
        for row, (pred, crys) in enumerate(((pb, cb), (pt, ct))):
            y = gi * 2 + row
            solid = row == 0
            ax.plot([pred, crys], [y, y], color="#B7B0A2", lw=0.7,
                    ls="-" if solid else ":", zorder=1)
            ax.plot(crys, y, marker="o", ms=2.6, mfc="white",
                    mec="#7A7672", mew=0.6, zorder=3)
            ax.plot(pred, y, marker="o", ms=2.6, mfc=col, mec="white",
                    mew=0.3, zorder=4)
            ax.text(pred - 1.0, y, f"{pred:.1f}", ha="right", va="center",
                    fontsize=4.8, zorder=5)
    ax.set_yticks([gi * 2 + 0.5 for gi in range(len(ms))])
    ax.set_yticklabels(ms)
    ax.invert_yaxis()
    ax.set_xlim(23.5, 79.5)
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.set_xlabel("Sequence recovery (%)")
    handles = [
        Line2D([], [], color="#B7B0A2", lw=0.7, ls="-",
               label="best sample"),
        Line2D([], [], color="#B7B0A2", lw=0.7, ls=":",
               label="t = 0.1 mean"),
    ]

    ax.legend(handles=handles, loc="lower right", fontsize=5.5,
              handlelength=1.6, handletextpad=0.4, borderaxespad=0.15,
              labelspacing=0.25)
    C.tighten_ylabels(fig)
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_g")
