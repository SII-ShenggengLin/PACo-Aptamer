"""Fig 6b | 8HZD_A 案例结构(PyMOL 渲染嵌入 + 格题/注记)。"""
import matplotlib
import matplotlib.image

import fig6_common as C
from fig_style import save

TITLE_FS, CAP_FS = 8.5, 7.5
TITLE_PAD, CAP_OFFSET = 3.5, 5.0
TITLE_BAND_IN = 0.187
CAP_BAND_IN = 0.173

def _cell_rect(png_path, col):
    """第 col 列渲染图的 axes 矩形(fig fraction):宽 = 列宽 × 0.75、列心"""
    img = matplotlib.image.imread(png_path)
    w_in = C.B_IMG_W_IN
    h_in = w_in * img.shape[0] / img.shape[1]
    x0_in = C.COLS[col] * C.FIG_W_IN + (C.CELL_IN - w_in) / 2
    slack = max(0.0, C.ROW2_H_IN - TITLE_BAND_IN - h_in - CAP_BAND_IN)
    top_in = C.ROW2_TOP_IN + slack / 2 + TITLE_BAND_IN
    return [x0_in / C.FIG_W_IN,
            (C.FIG_H_IN - top_in - h_in) / C.FIG_H_IN,
            w_in / C.FIG_W_IN, h_in / C.FIG_H_IN]

def _draw_cell(fig, Ldict, key, png_path, col, title, caption):
    rect = _cell_rect(png_path, col)
    Ldict[key] = rect
    ax = fig.add_axes(rect)
    img = matplotlib.image.imread(png_path)
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    ax.set_title(title, fontsize=TITLE_FS, pad=TITLE_PAD)
    ax.annotate(caption, xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -CAP_OFFSET), textcoords="offset points",
                ha="center", va="top", fontsize=CAP_FS)
    return ax

def draw(fig, L):
    ax1 = _draw_cell(fig, L, "b1", C.PYMOL_OVERVIEW, 0,
                     "8HZD_A \u00b7 N618", "7 interface residues (<5 \u00c5)")
    C.panel_label_at(fig, "b", C.IMG_X[0], C.ROW2_TOP_IN, size=10)
    _draw_cell(fig, L, "b2", C.PYMOL_POCKET, 1,
               "pocket close-up (<8 \u00c5)",
               "interface residues 7/8/14/15/28/29/30")
    return ax1

if __name__ == "__main__":
    fig, _L = C.make_canvas()
    draw(fig, _L)
    save(fig, "fig6_b")
    print("iface residues:", C.IFACE)
