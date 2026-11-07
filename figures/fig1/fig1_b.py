"""Fig 1b | GVP encoder 细节(对应 a 中同名模块)。"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

import fig1_common as C
from fig1_common import fbox, featbar, opath, txt

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY

def dash_varrow(ax, x, y_from, y_tip, hl=0.19, hw=0.066, lw=0.9,
                color=C.C_EDGE):
    """竖直虚线箭头:Line2D 虚杆 + 显式等腰三角头。"""
    d = -1 if y_tip < y_from else 1
    y_base = y_tip - d * hl
    ax.add_line(Line2D([x, x], [y_from, y_base], lw=lw, color=color,
                       zorder=3, linestyle=(0, (3, 2)),
                       solid_capstyle="round"))
    ax.add_patch(Polygon([(x, y_tip), (x - hw, y_base), (x + hw, y_base)],
                         closed=True, fc=color, ec="none", zorder=3))

def draw(ax):

    C.panel_frame(ax, 0.24, 0.24, 9.805, 9.52, "GVP encoder ×3",
                  fs=C.FS_HEAD, bh=0.84)

    featbar(ax, 1.00, 6.40, 0.50, 1.50, fc=C.C_FEAT)
    txt(ax, 1.25, 6.10, "Scalar", fs=SM)
    featbar(ax, 1.00, 3.70, 0.50, 1.50, fc=C.C_VEC)
    txt(ax, 1.25, 3.40, "Vector", fs=SM)
    fbox(ax, 0.50, 1.40, 2.30, 0.55)
    txt(ax, 1.65, 1.675, "Edge features", fs=SM)

    fbox(ax, 3.30, 1.75, 4.03, 7.00)
    txt(ax, 5.315, 8.40, "GVP layer ×3", fs=FS)
    fbox(ax, 3.80, 6.05, 3.25, 0.85)
    txt(ax, 5.425, 6.475, "Message MLP", fs=SM)
    fbox(ax, 3.80, 3.05, 3.25, 0.85)
    txt(ax, 5.425, 3.475, "Linear (no bias)", fs=SM)

    dash_varrow(ax, 5.60, 3.90, 6.03)
    txt(ax, 5.42, 4.95, "Norm", fs=TINY, ha="right")

    ax.add_patch(Circle((8.05, 6.47), 0.05, fc="black", ec="none", zorder=6))
    dash_varrow(ax, 8.05, 6.47, 5.25)

    ax.add_patch(Circle((8.05, 4.97), 0.24, fc="white", ec="black", lw=0.7,
                        zorder=5))
    txt(ax, 8.05, 4.97, "σ", fs=C.FS_TXT, z=6)
    dash_varrow(ax, 8.05, 4.73, 3.51)

    opath(ax, [(1.50, 7.15), (2.75, 7.15), (2.75, 6.47), (3.80, 6.47)])
    opath(ax, [(1.50, 4.45), (2.95, 4.45), (2.95, 3.47), (3.80, 3.47)])
    opath(ax, [(2.80, 1.67), (3.10, 1.67), (3.10, 6.30), (3.80, 6.30)])

    opath(ax, [(7.05, 6.47), (8.85, 6.47)], capstyle="butt")
    opath(ax, [(7.05, 3.47), (8.85, 3.47)], capstyle="butt")
    featbar(ax, 8.85, 5.72, 0.50, 1.50, fc=C.C_FEAT)
    txt(ax, 9.10, 5.42, "New scalar", fs=SM)
    featbar(ax, 8.85, 2.72, 0.50, 1.50, fc=C.C_VEC)
    txt(ax, 9.10, 2.42, "New vector", fs=SM)

    txt(ax, 4.79, 0.75, "Message passing over kNN + SS-pair edges",
        fs=C.FS_TXT)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes([C.RECT_BX[0], C.RECT_BCD_Y[0],
                     C.RECT_BCD_WB, C.RECT_BCD_Y[1]])
    ax.set_xlim(C.D_LIM_B[0], C.D_LIM_B[1])
    ax.set_ylim(C.D_LIM_B[2], C.D_LIM_B[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_b")
