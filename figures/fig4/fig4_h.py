"""Fig 4h | recovery-matched 负对照逐链 scTM 差分布:scTM(PACo) − scTM(control)。"""
import numpy as np
from matplotlib.lines import Line2D

import fig4_common as C
from fig_style import PACO, panel_label, save

def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "h")
    diff = np.array([v["sctm_PACo"] - v["sctm_randmatch"]
                     for v in C.SCTM["per_chain"].values()
                     if "sctm_PACo" in v and "sctm_randmatch" in v]) * 100
    ax.hist(diff, bins=36, color=PACO, alpha=0.85, ec="white", lw=0.2)
    ax.axvline(0, color="#7A7672", lw=0.6, ls=":")
    ax.axvline(float(np.mean(diff)), color="black", lw=0.8, ls="--")
    ax.legend(handles=[Line2D([], [], color="#7A7672", lw=0.6, ls=":",
                              label="zero"),
                       Line2D([], [], color="black", lw=0.8, ls="--",
                              label="mean")],
              loc="upper left", fontsize=6.0, handlelength=1.6,
              handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xlabel("Advantage over recovery-matched\nrandom control (%)")
    ax.set_ylabel("Chains")
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_h")
