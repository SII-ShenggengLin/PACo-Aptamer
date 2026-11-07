"""Fig 5a | 温度扫描 recovery–diversity 前沿(PACo-Aptamer / RhoDesign / gRNAde)。"""
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
