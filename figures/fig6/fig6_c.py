"""Fig 6c | 逐位恢复序列条(8HZD_A:native + 5 方法最优设计)。"""
import numpy as np
import matplotlib.pyplot as plt

import fig6_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save

FS_LETTER, FS_LABEL, FS_VAL = 7.0, 7.5, 7.5
FS_HEADER = 7.0
COL_ID, COL_IF = 39.4, 47.4

def draw(fig, L):
    axb = fig.add_axes(L["c"])
    panel_label(axb, "c", size=10)
    rows = [("native", C.natA, None, None)] + [
        (m, C.SEQS[m][0], C.SEQS[m][1], None) for m in METHOD_ORDER]
    L = len(C.natA)
    iface_pos = [C.RESIDS.index(rn) for rn in C.IFACE]

    def iface_id(seq):
        return np.mean([seq[p] == C.natA[p] for p in iface_pos])

    for ri, (name, sq, bv, _) in enumerate(rows):
        y0 = len(rows) - 1 - ri
        if name == "native":
            for pi, ch_ in enumerate(sq):
                axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                         fontsize=FS_LETTER, color="black")
            axb.text(-0.7, y0 + 0.5, "native", ha="right", va="center",
                     fontsize=FS_LABEL, color="black")
            continue
        col = METHOD_COLOR[name]
        for pi, ch_ in enumerate(sq):
            match = ch_ == C.natA[pi]
            axb.add_patch(plt.Rectangle((pi, y0), 1, 1,
                                        fc=col if match else "white",
                                        ec="#D3B5B0", lw=0.3))
            axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                     fontsize=FS_LETTER, color="black")
        axb.text(-0.7, y0 + 0.5, name, ha="right", va="center",
                 fontsize=FS_LABEL, color="black")
        fi = iface_id(sq)
        axb.text(COL_ID, y0 + 0.5, f"{100*bv:.0f}%", ha="center", va="center",
                 fontsize=FS_VAL, color="black")
        axb.text(COL_IF, y0 + 0.5, f"{100*fi:.0f}%", ha="center", va="center",
                 fontsize=FS_VAL, color="black")
    axb.text(COL_ID, 6.45, "identity", ha="center", va="bottom",
             fontsize=FS_HEADER, color="black")
    axb.text(COL_IF, 6.45, "interface", ha="center", va="bottom",
             fontsize=FS_HEADER, color="black")
    axb.set_xlim(-13.5, 52.4)

    axb.set_ylim(-0.45, 7.35)
    axb.axis("off")
    return axb

if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_c")
    for m in METHOD_ORDER:
        s, v = C.SEQS[m]
        ii = np.mean([s[p] == C.natA[p] for p in
                      [C.RESIDS.index(rn) for rn in C.IFACE]])
        print(f"{m:14s} overall {100*v:5.1f}%  iface {100*ii:5.1f}%")
