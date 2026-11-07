"""Supplementary Information 图公共模块(画布几何 + 数据加载 + 统计标注格式)。"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (
    ALIGNIF, AUX_CORAL, AUX_CYAN, AUX_ORANGE, AUX_PERIW, AUX_SAGE, AUX_SAND,
    DATA_DIR, GRNADE, LIGAND, MM, NAMPNN, NATIVE, NEG, RHODESIGN, PACO, PACO_D,
    apply_style, load,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUR = os.path.join(DATA_DIR, "results")
OUT_SI = os.path.abspath(os.environ.get(
    "PACO_OUTPUT_DIR", os.path.join(os.path.dirname(ROOT), "figures", "out")
))

METHODS = ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
MCOLOR = {"PACoAptamer": PACO, "PACo": PACO, "RhoDesign": RHODESIGN,
          "gRNAde": GRNADE, "NA-MPNN": NAMPNN, "AlignIF": ALIGNIF,
          "native": NATIVE, "randmatch": NEG}

MLAB = {"PACoAptamer": "PACo-Aptamer", "PACo": "PACo-Aptamer",
        "RhoDesign": "RhoDesign", "gRNAde": "gRNAde", "NA-MPNN": "NA-MPNN",
        "AlignIF": "AlignIF", "native": "native", "randmatch": "randmatch"}

def make_canvas(h_mm, nrows, ncols, **gs_kw):
    """183mm 全宽画布;h_mm 高度;返回 (fig, gs)。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(183 * MM, h_mm * MM))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gs_kw)
    return fig, gs

def save_si(fig, slug, dpi=600):
    import matplotlib.pyplot as plt
    os.makedirs(OUT_SI, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_SI, f"si_{slug}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", f"si_{slug}")

def load_csv(rel):
    with open(os.path.join(SUR, rel), newline="") as f:
        return list(csv.DictReader(f))

def pfmt_sci(p):
    """P 值格式:≥0.05 → P = 0.42;0.01–0.05 → P = 0.022;<0.01 → P = 5.4×10⁻⁵。"""
    if p is None:
        return "P = n.a."
    if p >= 0.05:
        return f"P = {p:.2f}"
    if p >= 0.01:
        return f"P = {p:.3f}"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10 ** exp
    return rf"$P = {mant:.1f}\times10^{{{exp}}}$"

def sig_star(p):
    if p is None:
        return "n.t."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
