"""Fig 2d | 逐链恢复率差值的累积分布(ECDF):PACo-Aptamer 减四条基线。"""
import numpy as np

import fig2_common as C
from fig_style import METHOD_COLOR, panel_label, save

OPPS = ["RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
XLIM = (-25, 72)

def draw(fig, gs):
    axd = fig.add_subplot(gs[1, 0])
    panel_label(axd, "d")
    for opp in OPPS:
        d = np.sort(np.array([C.pc["PACo-Aptamer"][s] - C.pc[opp][s]
                              for s in C.STEMS]) * 100)
        axd.step(np.concatenate([d, [XLIM[1]]]),
                 np.arange(1, len(d) + 2) / (len(d) + 1),
                 where="post", color=METHOD_COLOR[opp], lw=1.0,
                 label=opp)
    axd.axvline(0, color="#7A7672", lw=0.6, ls="--")
    axd.set_xlim(*XLIM); axd.set_ylim(0, 1)
    axd.set_xticks([-20, 0, 20, 40, 60])
    axd.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axd.set_yticklabels(["0", "25", "50", "75", "100"])
    axd.set_xlabel("Recovery advantage of PACo-Aptamer\n"
                   "over each baseline (%)")
    axd.set_ylabel("Cumulative share of chains (%)")
    axd.legend(loc="lower right", fontsize=4.6, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    return axd

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_d")
