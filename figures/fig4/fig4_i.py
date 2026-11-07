"""Fig 4i | 碱基对接触保真 scMCC 点距图:点 = 均值,须 = 95% CI。"""
import numpy as np

import fig4_common as C
from fig_style import METHOD_COLOR, NATIVE, NEG, panel_label, save

ROWS = [("native", "native", NATIVE),
        ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"]),
        ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
        ("PACo-Aptamer", "PACo", METHOD_COLOR["PACo-Aptamer"]),
        ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
        ("random\n(recovery-matched)", "randmatch", NEG)]

def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 2])
    panel_label(ax, "i")
    ylabs = []
    for i, (lab, key, col) in enumerate(ROWS):
        s = C.SCMCC["summary"][key]
        m, lo, hi = s["scMCC_mean"], s["scMCC_ci95"][0], s["scMCC_ci95"][1]
        yp = len(ROWS) - 1 - i
        ax.errorbar(m, yp, xerr=[[m - lo], [hi - m]], fmt="o", ms=3.6,
                    mfc=col, mec="white", mew=0.4, lw=0, elinewidth=0.7,
                    capsize=1.4, ecolor="#7A7672", zorder=3)
        ax.text(hi + 0.028, yp, f"{m:.3f}", ha="left", va="center",
                fontsize=5.2)
        ylabs.append(lab)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ylabs[::-1], fontsize=5.0)
    ax.set_xlim(0, 0.78)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xlabel("scMCC")
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_i")
