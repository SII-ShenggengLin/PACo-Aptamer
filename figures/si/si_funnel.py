"""Supplementary Fig. 1 | 数据漏斗与三层测试集(数据集如何划分)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import si_common as C
EDGE = "black"

FS, FS_S = 6.0, 5.2
PAD = 1.0
TIP = 0.0

def wbox(ax, x, y, w, h, lw=0.9, dashed=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}",
                                fc="white", ec=EDGE, lw=lw,
                                linestyle=(0, (3, 3)) if dashed else "-"))

def arrow(ax, x0, y0, x1, y1, lw=0.8):
    """横平竖直箭头(x0==x1 垂直,y0==y1 水平),不做任何弯曲。"""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=EDGE, lw=lw,
                                 shrinkA=0, shrinkB=0))

def lab(ax, x, y, txt, fs=FS, ha="center"):
    ax.text(x, y, txt, fontsize=fs, color="black", ha=ha, va="center",
            linespacing=1.35)

def main():

    fig, gs = C.make_canvas(104, 1, 1, left=0.0, right=1.0, top=1.0,
                            bottom=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.5, 182.5); ax.set_ylim(29.0, 133.0)
    ax.set_aspect("equal"); ax.axis("off")

    aB, bB, cB, dB, eB, fB = 120.5, 103.0, 85.5, 63.4, 48.4, 34.4
    aT, bT, cT, dT, eT, fT = 130.0, 112.5, 95.0, 72.9, 57.4, 43.4
    y_bus = 80.5

    x_center = 91.5
    w_left, w_right, top_gap = 37.0, 42.0, 22.0
    x_left = x_center - (w_left + top_gap + w_right) / 2
    x_right = x_left + w_left + top_gap
    xcL, xcR = x_left + w_left / 2, x_right + w_right / 2
    w_cd = 57.0
    xcd = x_center
    x_cd = xcd - w_cd / 2

    wbox(ax, x_left, aB, w_left, aT - aB)
    lab(ax, xcL, (aB + aT) / 2, "PDB entries containing RNA\n24,252")
    wbox(ax, x_right, aB, w_right, aT - aB)
    lab(ax, xcR, (aB + aT) / 2, "RNA\u2013ligand complexes\n868")
    arrow(ax, x_left + w_left + PAD, (aB + aT) / 2,
          x_right - PAD - TIP, (aB + aT) / 2)

    arrow(ax, xcL, aB - PAD, xcL, bT + PAD + TIP)
    wbox(ax, x_left, bB, w_left, bT - bB)
    lab(ax, xcL, (bB + bT) / 2, "RNA-only pool\n6,988 chains")

    arrow(ax, xcR, aB - PAD, xcR, bT + PAD + TIP)
    wbox(ax, x_right, bB, w_right, bT - bB)
    lab(ax, xcR, (bB + bT) / 2, "stage-2 finetuning pool\n"
                                "768 chains (510 entries)")
    lab(ax, xcR - 3, (aB - PAD + bT + PAD) / 2,
        "complexes sharing a cluster with\nany test chain are excluded",
        FS_S, ha="right")

    y_elbow = (bB - PAD + cT + PAD) / 2
    ax.plot([xcL, xcL, xcd], [bB - PAD, y_elbow, y_elbow],
            color=EDGE, lw=0.8)
    arrow(ax, xcd, y_elbow, xcd, cT + PAD + TIP)
    wbox(ax, x_cd, cB, w_cd, cT - cB)
    lab(ax, xcd, (cB + cT) / 2, "CD-HIT 0.8 + TM \u2265 0.45\ntransitive "
                                "closure \u2192 300 equivalence classes")

    ax.plot([xcd, xcd], [cB - PAD, y_bus], color=EDGE, lw=0.6)
    ax.plot([19.5, 163.5], [y_bus, y_bus], color=EDGE, lw=0.6)
    lab(ax, xcd + 13.5, (y_bus + cB - PAD) / 2,
        "class-level 8:1:1 split", FS_S)

    ww = 33.0
    splits = [(3, "train", "5,773", False),
              (39, "validation", "96", False),
              (75, "test", "477", False),
              (111, "independent", "235", False),
              (147, "unassigned", "511", True)]
    for x0, nm, cnt, dashed in splits:
        wbox(ax, x0, dB, ww, dT - dB, dashed=dashed)
        lab(ax, x0 + ww / 2, dB + 6.5, nm)
        lab(ax, x0 + ww / 2, dB + 3.0, cnt)
        arrow(ax, x0 + ww / 2, y_bus, x0 + ww / 2, dT + PAD + TIP, lw=0.7)
    lab(ax, 3, 59.0, "stage-1 pretraining\n(identical to baseline "
                     "retraining)", FS_S, ha="left")

    arrow(ax, 91.5, dB - PAD, 91.5, eT + PAD + TIP)
    wbox(ax, 75.5, eB, 32, eT - eB, lw=1.1)
    lab(ax, 91.5, eB + 6.2, "layer 1")
    lab(ax, 91.5, eB + 2.7, "cluster-based holdout")

    arrow(ax, 127.5, dB - PAD, 127.5, eT + PAD + TIP)
    wbox(ax, 111.5, eB, 32, eT - eB, lw=1.1)
    lab(ax, 127.5, eB + 6.2, "layer 2")
    lab(ax, 127.5, eB + 2.7, "time-based split (2024+)")

    arrow(ax, 101.5, eB - PAD, 101.5, fT + PAD + TIP, lw=0.9)
    arrow(ax, 117.5, eB - PAD, 117.5, fT + PAD + TIP, lw=0.9)
    lab(ax, 99.5, (eB - PAD + fT + PAD) / 2, "54", FS_S, ha="right")
    lab(ax, 119.5, (eB - PAD + fT + PAD) / 2, "39", FS_S, ha="left")
    wbox(ax, 86, fB, 47, fT - fB, lw=1.1)
    lab(ax, 109.5, fB + 6.2, "layer 3")
    lab(ax, 109.5, fB + 2.7, "ligand holdout (zero-shot), 93 chains")
    lab(ax, 109.5, 31.4, "ligands absent from all training complexes", FS_S)
    return fig

if __name__ == "__main__":
    fig = main()
    C.save_si(fig, "funnel")
