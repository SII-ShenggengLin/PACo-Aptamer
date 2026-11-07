"""Fig 1a | 架构总览:条件输入 → 编码 → 联合注意力 → 自回归解码。"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fig1_common as C
from fig1_common import fbox, img_place, opath, txt, featbar

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY

COLS = ((0.35, 8.30, "Conditioning inputs"), (9.35, 8.30, "Encoders"),
        (18.35, 7.60, "Joint RNA–ligand attention"),
        (26.65, 6.00, "Autoregressive decoder"))

def draw(ax):

    for x, w, t in COLS:
        C.panel_frame(ax, x, 0.36, w, 12.86, t, fs=C.FS_HEAD, bh=1.00)

    _inputs(ax)
    _encoders(ax)
    _joint(ax)
    _decoder(ax)

    for y in (11.01, 7.22, 2.62):
        opath(ax, [(8.65, y), (9.35, y)], lw=0.8)
    for y in (10.09, 1.73):
        opath(ax, [(17.65, y), (18.35, y)], lw=0.8)
    opath(ax, [(25.95, 6.72), (26.65, 6.72)], lw=0.8)

def _inputs(ax):

    fbox(ax, 0.57, 9.79, 7.86, 2.43)
    img_place(ax, f"{C.ASSETS}/fig1_bb.png", 2.70, 11.42, h=1.36)
    img_place(ax, f"{C.ASSETS}/fig1_nt.png", 6.00, 11.42, h=1.36)
    txt(ax, 2.70, 10.35, "RNA backbone", fs=SM)
    txt(ax, 6.00, 10.35, "9 atoms / nt", fs=SM)

    fbox(ax, 0.57, 5.195, 7.86, 4.05)
    txt(ax, 4.50, 8.71, "Predicted secondary structure", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_ss.png", 2.591, 6.90, h=2.55)
    opath(ax, [(4.531, 6.90), (5.369, 6.90)], lw=0.8)
    pairs_ss, _, resis, seq = C.load_pairs()
    C.pairmap_glyph(ax, 6.809, 6.90, 0.040, pairs_ss, resis, seq)

    fbox(ax, 0.57, 0.58, 7.86, 4.07)
    txt(ax, 4.50, 4.26, "Ligand pocket atoms", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_pocket.png", 4.50, 2.70, h=2.52)
    txt(ax, 4.50, 0.94, "Pocket residues + ligand sticks", fs=SM)

def _encoders(ax):

    fbox(ax, 9.57, 7.96, 7.86, 4.26)

    txt(ax, 13.50, 11.60, "GVP encoder ×3", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_graph.png", 11.54, 10.03, h=1.90)
    txt(ax, 11.54, 8.68, "Geometric graph", fs=SM)

    opath(ax, [(13.19, 10.03), (13.78, 10.03)], lw=0.8)
    fbox(ax, 13.78, 9.48, 2.20, 1.10)
    txt(ax, 14.88, 10.25, "GVP layer", fs=SM)
    txt(ax, 14.88, 9.81, "×3", fs=SM)

    opath(ax, [(15.98, 10.48), (16.52, 10.48)], lw=0.8)
    opath(ax, [(15.98, 9.58), (16.52, 9.58)], lw=0.8)
    featbar(ax, 16.52, 10.13, 0.30, 0.70, fc=C.C_FEAT)
    featbar(ax, 16.52, 9.23, 0.30, 0.70, fc=C.C_VEC)
    txt(ax, 16.96, 10.48, "$\\hat{s}$", fs=SM, ha="left")
    txt(ax, 16.96, 9.58, "$\\hat{v}$", fs=SM, ha="left")

    fbox(ax, 9.57, 0.58, 7.86, 6.86)
    txt(ax, 13.50, 6.56, "Ligand atom embedding", fs=FS)
    fbox(ax, 10.90, 5.24, 5.20, 0.72)
    txt(ax, 13.50, 5.60, "Ligand heavy atoms", fs=SM)
    opath(ax, [(12.30, 5.24), (12.30, 4.84)], lw=0.8)
    opath(ax, [(14.70, 5.24), (14.70, 4.84)], lw=0.8)
    fbox(ax, 9.98, 3.72, 3.40, 1.12)
    txt(ax, 11.68, 4.49, "Element", fs=SM)
    txt(ax, 11.68, 4.05, "one-hot (5)", fs=SM)
    fbox(ax, 13.62, 3.72, 3.40, 1.12)
    txt(ax, 15.32, 4.49, "Distance to RNA", fs=SM)
    txt(ax, 15.32, 4.05, "RBF ×16", fs=SM)
    opath(ax, [(12.30, 3.72), (12.30, 3.32)], lw=0.8)
    opath(ax, [(14.70, 3.72), (14.70, 3.32)], lw=0.8)
    fbox(ax, 10.90, 2.60, 5.20, 0.72)
    txt(ax, 13.50, 2.96, "2-layer MLP → 512-d", fs=SM)
    opath(ax, [(13.50, 2.60), (13.50, 2.24)], lw=0.8)

    featbar(ax, 13.25, 1.22, 0.50, 1.02, fc=C.C_VEC)

def _joint(ax):

    fbox(ax, 18.90, 0.73, 6.50, 11.39)

    txt(ax, 22.15, 11.60, "RNA + ligand tokens", fs=FS)

    n_rna, n_lig, cell = 4, 3, 0.64
    n = n_rna + n_lig
    cx = 22.15
    x0, y1 = cx - n * cell / 2, 11.08
    for i in range(n):
        for j in range(n):
            in_rna = i < n_rna and j < n_rna
            in_lig = i >= n_rna and j >= n_rna
            fc = C.C_VEC if in_rna else ("#E8CFC4" if in_lig else C.C_WHITE)
            ax.add_patch(Rectangle((x0 + j * cell, y1 - (i + 1) * cell),
                                   cell, cell, fc=fc, ec="white", lw=0.5,
                                   zorder=2))
    ax.plot([x0, x0 + n * cell], [y1 - n_rna * cell] * 2, color=C.C_TEXT,
            lw=1.0, zorder=3)
    ax.plot([x0 + n_rna * cell] * 2, [y1 - n * cell, y1], color=C.C_TEXT,
            lw=1.0, zorder=3)
    ax.add_patch(Rectangle((x0, y1 - n * cell), n * cell, n * cell,
                           fc="none", ec=C.C_EDGE, lw=0.9, zorder=3))

    txt(ax, x0 + 2 * cell, y1 - 2 * cell, "RNA", fs=SM, color=C.C_TEXT,
        weight="bold", z=4)
    txt(ax, x0 + 5.5 * cell, y1 - 5.5 * cell, "Lig", fs=SM, color=C.C_TEXT,
        weight="bold", z=4)

    opath(ax, [(cx, 6.60), (cx, 6.05)], lw=0.8)

    fbox(ax, 19.91, 4.55, 4.48, 1.50)
    txt(ax, cx, 5.56, "Joint attention ×3", fs=FS)
    txt(ax, cx, 5.04, "512-d · 4 heads", fs=SM)
    opath(ax, [(cx, 4.47), (cx, 4.07)], lw=0.8)
    fbox(ax, 19.91, 2.82, 4.48, 1.25)
    txt(ax, cx, 3.44, "$p(\\hat{s}_t \\mid s_{<t}, \\mathrm{RNA}, "
                      "\\mathrm{ligand})$", fs=SM)
    txt(ax, cx, 2.33, "Joint states passed to", fs=C.FS_TXT)
    txt(ax, cx, 1.85, "decoder cross-attention", fs=C.FS_TXT)

    ax.add_patch(Rectangle((20.89, 1.19), 0.24, 0.24, fc=C.C_VEC,
                           ec=C.C_EDGE, lw=0.5, zorder=3))
    txt(ax, 21.23, 1.31, "RNA", fs=TINY, ha="left")
    ax.add_patch(Rectangle((22.25, 1.19), 0.24, 0.24, fc="#E8CFC4",
                           ec=C.C_EDGE, lw=0.5, zorder=3))
    txt(ax, 22.59, 1.31, "ligand", fs=TINY, ha="left")

def _decoder(ax):

    cx = 29.65

    fbox(ax, 27.40, 1.28, 4.50, 0.95)
    txt(ax, cx, 1.755, "Partial sequence $s_{<t}$", fs=SM)
    opath(ax, [(cx, 2.28), (cx, 3.275)], lw=0.8, capstyle="butt")

    fbox(ax, 27.40, 3.275, 4.50, 2.70)
    txt(ax, cx, 5.355, "Decoder layer ×3", fs=SM)
    txt(ax, cx, 4.625, "Masked self-attn ·", fs=SM)
    txt(ax, cx, 3.895, "Cross-attn · FFN", fs=SM)
    opath(ax, [(cx, 6.025), (cx, 7.02)], lw=0.8, capstyle="butt")
    fbox(ax, 27.40, 7.02, 4.50, 0.95)
    txt(ax, cx, 7.495, "Softmax $p(\\hat{s}_t)$", fs=SM)
    opath(ax, [(cx, 8.02), (cx, 9.015)], lw=0.8, capstyle="butt")
    fbox(ax, 27.40, 9.015, 4.50, 0.90)
    txt(ax, cx, 9.465, "Sample $\\hat{s}_t$", fs=SM)

    opath(ax, [(cx, 9.965), (cx, 10.96)], lw=0.8, capstyle="butt")

    seq = "AUGGCUAGCUAA"
    for k, ch in enumerate(seq):
        x0 = 27.03 + k * 0.44
        ax.add_patch(Rectangle((x0, 11.06), 0.40, 0.45,
                               fc=C.NT_COLORS[ch], ec="white", lw=0.5,
                               zorder=3))
        txt(ax, x0 + 0.20, 11.285, ch, fs=SM, color="white", weight="bold",
            z=4)
    txt(ax, 29.65, 11.93, "RNA sequence", fs=SM)

    opath(ax, [(31.90, 9.465), (32.25, 9.465), (32.25, 1.755), (31.90, 1.755)],
          lw=0.8)

if __name__ == "__main__":
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes(C.RECT_A)
    ax.set_xlim(C.A_LIM[0], C.A_LIM[1])
    ax.set_ylim(C.A_LIM[2], C.A_LIM[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_a")
