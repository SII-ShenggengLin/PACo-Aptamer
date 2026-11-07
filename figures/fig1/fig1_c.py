"""Fig 1c | Transformer encoder block 细节(对应 a 中联合注意力模块)。"""
from matplotlib.patches import Circle

import fig1_common as C
from fig1_common import fbox, opath, plus, txt

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY

def draw(ax):

    C.panel_frame(ax, 0.24, 0.24, 7.35, 9.52, "Transformer encoder ×3",
                  fs=C.FS_HEAD, bh=0.84)
    cx = 3.665

    fbox(ax, 1.165, 1.25, 5.00, 0.60)
    txt(ax, cx, 1.55, "Nucleotide + ligand tokens", fs=SM)

    opath(ax, [(cx, 1.85), (cx, 2.60)])
    fbox(ax, 1.615, 2.60, 4.10, 0.80)
    txt(ax, cx, 3.15, "LayerNorm + multi-head", fs=SM)
    txt(ax, cx, 2.85, "self-attention", fs=SM)
    opath(ax, [(cx, 3.40), (cx, 4.15)], ms=4.5)
    plus(ax, cx, 4.29)

    ax.add_patch(Circle((cx, 2.225), 0.05, fc="black", ec="none", zorder=6))
    ax.plot([cx, 1.40, 1.40, 3.525], [2.225, 2.225, 4.29, 4.29],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    opath(ax, [(cx, 4.43), (cx, 5.18)])
    fbox(ax, 1.615, 5.18, 4.10, 0.70)
    txt(ax, cx, 5.53, "LayerNorm + FFN", fs=SM)
    opath(ax, [(cx, 5.88), (cx, 6.63)], ms=4.5)
    plus(ax, cx, 6.77)

    ax.add_patch(Circle((cx, 4.805), 0.05, fc="black", ec="none", zorder=6))
    ax.plot([cx, 0.90, 0.90, 3.525], [4.805, 4.805, 6.77, 6.77],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    opath(ax, [(cx, 6.91), (cx, 7.66)])
    fbox(ax, 2.015, 7.66, 3.30, 0.60)
    txt(ax, cx, 7.96, "Updated tokens", fs=SM)

    opath(ax, [(5.315, 7.96), (6.465, 7.96), (6.465, 3.00), (5.715, 3.00)],
          ls=(0, (4, 3)), color=C.C_DASH, ms=5.0)
    txt(ax, 6.615, 5.48, "×3", fs=SM, ha="left")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes([C.RECT_BX[1], C.RECT_BCD_Y[0],
                     C.RECT_BCD_WC, C.RECT_BCD_Y[1]])
    ax.set_xlim(C.D_LIM_C[0], C.D_LIM_C[1])
    ax.set_ylim(C.D_LIM_C[2], C.D_LIM_C[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_c")
