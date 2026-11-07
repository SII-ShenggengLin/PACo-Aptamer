"""Fig 6d | 适配体 vs 结构同条件随机对照的 Δ log probability 分布(双库)。"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig6_common as C
from fig_style import NEG, save

def draw(fig, L):
    axd = fig.add_axes(L["d"])
    C.panel_label_at(fig, "d", C.IMG_X[0], C.ROW3_TOP_IN)
    bins = np.linspace(0.0, 4.0, 41)
    axd.hist(C.D_APT, bins=bins, color="#8FA6BB", alpha=0.85, ec="white",
             lw=0.15)
    axd.hist(C.D_UTX, bins=bins, color="#D9C3A9", alpha=0.75, ec="white",
             lw=0.15)
    axd.axvline(np.mean(C.D_ALL), color=NEG, lw=0.9, ls="--")
    handles = [
        Patch(fc="#8FA6BB", ec="none", label=f"AptaDB (n = {len(C.D_APT)})"),
        Patch(fc="#D9C3A9", ec="none", label=f"UTexas (n = {len(C.D_UTX)})"),
        Line2D([], [], color=NEG, lw=0.9, ls="--",
               label=f"merged mean {np.mean(C.D_ALL):+.2f}"),
    ]
    axd.legend(handles=handles, loc="upper left", frameon=True,
               facecolor="white", framealpha=1.0, edgecolor="none",
               handlelength=1.4, borderaxespad=0.4)
    axd.set_xlabel("\u0394 log probability per interaction")
    axd.set_ylabel("Interactions")
    axd.set_xlim(0.0, 4.0)

    axd.set_ylim(0.0, 26.0)
    return axd

if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_d")
