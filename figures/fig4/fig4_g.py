"""Fig 4g | 自洽结构 scTM 水平箱线:native 上界 + 4 方法 + recovery-matched 负对照。"""
import numpy as np
from matplotlib.lines import Line2D

import fig4_common as C
from fig_style import METHOD_COLOR, NATIVE, NEG, panel_label, save

ROWS = [("native", "native", NATIVE),
        ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
        ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"]),
        ("PACo-Aptamer", "PACo", METHOD_COLOR["PACo-Aptamer"]),
        ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
        ("random\n(recovery-matched)", "randmatch", NEG)]

def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "g")
    ylabs = []
    for i, (lab, key, col) in enumerate(ROWS):
        vals = np.array([v[f"sctm_{key}"] for v in C.SCTM["per_chain"].values()
                         if f"sctm_{key}" in v])
        yp = len(ROWS) - 1 - i
        bp = ax.boxplot([vals], positions=[yp], vert=False, widths=0.55,
                        whis=1.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=0.7),
                        boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                        capprops=dict(lw=0.4))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.85)
        ax.scatter([np.mean(vals)], [yp], color="black", s=5, lw=0, zorder=5)
        sm = C.SCTM["summary"][key]
        w_hi = bp["whiskers"][1].get_xdata()[1]
        ax.text(w_hi + 0.03, yp, f"{sm['mean']:.3f} · "
                f"{100*sm['pass_045']:.0f}%", ha="left", va="center",
                fontsize=5.2)
        ylabs.append(lab)
    ax.axvline(0.45, color="#7A7672", lw=0.5, ls=":")

    ax.legend(handles=[Line2D([], [], color="#7A7672", lw=0.5, ls=":",
                              label="scTM = 0.45")],
              loc="upper right", fontsize=6.0, handlelength=1.6,
              handletextpad=0.5, borderaxespad=0.2)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ylabs[::-1], fontsize=5.0)
    ax.set_ylim(-0.55, 6.0)
    ax.set_xlabel("scTM (RhoFold+ refold)")
    ax.set_xlim(0, 1.24)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_g")
