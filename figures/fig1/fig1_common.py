"""Shared styling, canvas and drawing primitives for Figure 1 panels."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(DIR))
DATA_DIR = os.path.abspath(os.environ.get("PACO_DATA_DIR", os.path.join(ROOT, "data")))
ASSETS = os.path.join(DATA_DIR, "figure_assets", "fig1")
OUTDIR = os.path.abspath(os.environ.get("PACO_OUTPUT_DIR", os.path.join(ROOT, "figures", "out")))
os.makedirs(OUTDIR, exist_ok=True)

C_FEAT = "#EAF2F8"
C_VEC = "#A9C4D9"
C_TERRA = "#C08B7C"
C_WHITE = "#FFFFFF"
C_TEXT = "#262626"
C_EDGE = "#333333"
C_DASH = "#9A9A9A"

NT_COLORS = {"A": "#A3B18A", "U": "#C08B7C", "G": "#8FA6BB", "C": "#D9C3A9"}

FIG_W_IN = 183 / 25.4
FIG_H_IN = 144 / 25.4
DPI = 400

FS_TITLE = 9.0
FS_HEAD = 7.5
FS_BOX = 7.0
FS_TXT = 6.5
FS_SM = 6.0
FS_TINY = 5.5

RECT_A = [0.002, 0.485, 0.996, 0.511]
RECT_BCD_Y = [0.010, 0.4390]
RECT_BX = [0.008, 0.386, 0.678]
RECT_BCD_WB = 0.360
RECT_BCD_WC = 0.274
RECT_BCD_WD = 0.311

A_LIM = (0.0, 33.0, 0.0, 13.32)
D_LIM = (0.0, 8.884, 0.0, 9.87)
D_LIM_B = (0.0, 10.285, 0.0, 9.87)
D_LIM_C = (0.0, 7.828, 0.0, 9.87)

def new_canvas():
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax_a = fig.add_axes(RECT_A)
    ax_a.set_xlim(A_LIM[0], A_LIM[1])
    ax_a.set_ylim(A_LIM[2], A_LIM[3])
    ax_a.axis("off")
    axs = [ax_a]
    for x0, w, lim in zip(RECT_BX, (RECT_BCD_WB, RECT_BCD_WC, RECT_BCD_WD),
                          (D_LIM_B, D_LIM_C, D_LIM)):
        ax = fig.add_axes([x0, RECT_BCD_Y[0], w, RECT_BCD_Y[1]])
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.axis("off")
        axs.append(ax)
    return fig, axs

def fbox(ax, x, y, w, h, fc=C_WHITE, ec=C_EDGE, lw=0.8, r=0.09, z=2, ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls,
                       mutation_aspect=1.0)
    ax.add_patch(b)
    return b

def txt(ax, x, y, s, fs=FS_TXT, ha="center", va="center", color=C_TEXT,
        weight="normal", z=6, style="normal", rotation=0):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color,
                   fontweight=weight, zorder=z, fontstyle=style,
                   rotation=rotation)

def opath(ax, pts, arrow=True, lw=0.9, color=C_EDGE, z=3, ls="-", ms=6.5,
          capstyle="round"):
    """Polyline through pts; arrow head at the last segment."""
    pts = list(pts)
    if len(pts) > 2:
        ax.add_line(Line2D([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                           lw=lw, color=color, zorder=z, solid_capstyle="round",
                           linestyle=ls))
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    if arrow:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=ms, lw=lw, color=color,
                                     zorder=z, linestyle=ls,
                                     shrinkA=0, shrinkB=0,
                                     capstyle=capstyle))
    else:
        ax.add_line(Line2D([x1, x2], [y1, y2], lw=lw, color=color, zorder=z,
                           solid_capstyle="round", linestyle=ls))

def plus(ax, x, y, r=0.14, lw=0.9, z=4):
    ax.add_patch(Circle((x, y), r, fc=C_WHITE, ec=C_TEXT, lw=lw, zorder=z))
    ax.add_line(Line2D([x - r * 0.55, x + r * 0.55], [y, y], lw=lw,
                       color=C_TEXT, zorder=z + 1))
    ax.add_line(Line2D([x, x], [y - r * 0.55, y + r * 0.55], lw=lw,
                       color=C_TEXT, zorder=z + 1))

def img_place(ax, path, cx, cy, h, z=3):
    """Place a (possibly transparent) PNG, auto-cropped to its content."""
    import PIL.Image
    im = PIL.Image.open(path)
    if im.mode == "RGBA":
        a = np.asarray(im.split()[3])
        ys, xs = np.where(a > 8)
        if len(xs):
            im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w0, h0 = im.size
    w = h * w0 / h0
    ax.imshow(im, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
              zorder=z, interpolation="antialiased", aspect="auto")
    return w

def panel_frame(ax, x, y, w, h, label, fs=FS_HEAD, bh=0.26):
    """Outer frame with a bold section label set into the top edge."""
    fig = ax.figure
    pos = ax.get_position()
    mm_per_unit = (pos.width * fig.get_figwidth() * 25.4
                   / (ax.get_xlim()[1] - ax.get_xlim()[0]))
    fbox(ax, x, y, w, h, fc="none", ec=C_EDGE, lw=1.0,
         r=2.0 / mm_per_unit, z=1)
    txt(ax, x + w / 2, y + h - bh / 2, label, fs=fs, weight="bold", z=4)

def panel_letter(ax, letter, size=9):
    """Bold panel letter just outside the axes' top-left corner."""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = fig.get_figwidth(), fig.get_figheight()
    dx = (1.2 / 25.4) / w_in
    dy = (1.2 / 25.4) / h_in
    fig.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=size,
             fontweight="bold", va="bottom", ha="left", color="black")

def save(fig, name, dpi=600):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)
    print("saved", name)

def featbar(ax, x, y, w, h, fc=C_FEAT, ec=C_EDGE, lw=0.8, z=3):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))

def load_pairs():
    with open(os.path.join(ASSETS, "fig1_1u8d.json")) as f:
        d = json.load(f)
    return d["pairs_ss"], d["pairs_knn"], d["resis"], d["seq"]

def pairmap_glyph(ax, cx, cy, cell, pairs, resis, seq, n=None,
                  hl_resi=None, hl_color=C_TERRA, fs=FS_TINY):
    """Adjacency (contact) map: backbone + SS pairs + sequence-letter ticks."""
    if n is None:
        n = len(seq)
    x0, y1 = cx - n * cell / 2, cy + n * cell / 2

    def cell_xy(i, j):
        return x0 + j * cell, y1 - (i + 1) * cell

    for i in range(n - 1):
        for (a, b) in ((i, i + 1), (i + 1, i)):
            ax.add_patch(Rectangle(cell_xy(a, b), cell, cell, fc="#DADADA",
                                   ec="none", zorder=2))
    for (i, j) in pairs:
        for (a, b) in ((i, j), (j, i)):
            ax.add_patch(Rectangle(cell_xy(a, b), cell, cell, fc=hl_color,
                                   ec="none", zorder=2))
    if hl_resi is not None and hl_resi in resis:
        k = resis.index(hl_resi)
        ax.add_patch(Rectangle((x0 + k * cell, y1 - n * cell), cell, n * cell,
                               fc="none", ec=C_TEXT, lw=0.8, zorder=4))
        ax.add_patch(Rectangle((x0, y1 - (k + 1) * cell), n * cell, cell,
                               fc="none", ec=C_TEXT, lw=0.8, zorder=4))
    ax.add_patch(Rectangle((x0, y1 - n * cell), n * cell, n * cell,
                           fc="none", ec=C_EDGE, lw=0.8, zorder=3))
