"""Fig 6e | 家族内 log probability–亲和力相关森林图(独立编号)。"""
import numpy as np

import fig6_common as C
from fig_style import PACO, save

def draw(fig, L):
    axe = fig.add_axes(L["e"])

    C.panel_label_at(fig, "e", C.COLS[2] * C.FIG_W_IN + 0.015, C.ROW3_TOP_IN)
    order = np.argsort(C.FAM_RHO)
    for i, oi in enumerate(order):
        col = "#7A7672" if C.FAM_RHO[oi] >= 0 else "#B5838D"
        axe.plot([min(C.FAM_RHO[oi], 0), max(C.FAM_RHO[oi], 0)], [i, i],
                 color=col, lw=1.3, solid_capstyle="round")
        axe.scatter([C.FAM_RHO[oi]], [i], s=2.5 + 1.1 * C.FAM_N[oi], c=col,
                    lw=0)
    axe.axvline(0, color="#7A7672", lw=0.6)
    axe.axvline(np.mean(C.FAM_RHO), color=PACO, lw=0.9, ls=":")

    axe.text(np.mean(C.FAM_RHO) + 0.05, len(C.FAM_RHO) - 1 + 2.3,
             f"mean \u03c1 = {np.mean(C.FAM_RHO):+.2f}",
             ha="left", va="center", fontsize=8.0, color="black")
    axe.set_yticks([])
    axe.set_xlabel("Spearman \u03c1")
    axe.set_xlim(-1.2, 1.2)
    axe.set_ylim(-1.3, len(C.FAM_RHO) - 1 + 3.6)
    return axe

if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_e")
    print("families:", len(C.FAM_RHO), "n range:",
          min(C.FAM_N), "-", max(C.FAM_N))
