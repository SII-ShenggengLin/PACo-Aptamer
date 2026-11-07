"""PACo-Aptamer 论文图表公共绘图规范(Nature Methods 落地)。"""
import json
import os

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.environ.get("PACO_DATA_DIR", os.path.join(ROOT, "data")))
SUR = os.path.join(DATA_DIR, "results")
FIGURE_ASSETS = os.path.join(DATA_DIR, "figure_assets")
OUT = os.path.abspath(os.environ.get("PACO_OUTPUT_DIR", os.path.join(ROOT, "figures", "out")))

PALETTE = ["#B5838D", "#D3B5B0", "#C08B7C", "#D9C3A9", "#CDBF9E", "#A3B18A",
           "#8DA290", "#8FA6BB", "#6B7B8D", "#A99AAF", "#B7B0A2", "#7A7672"]

PACO = "#6B7B8D"
PACO_D = "#8FA6BB"
RHODESIGN = "#7A7672"
ALIGNIF = "#A99AAF"
NAMPNN = "#B7B0A2"
GRNADE = "#CDBF9E"
NATIVE = "#A3B18A"
NEG = "#B5838D"
GRAY = "#7A7672"
LIGAND = "#C08B7C"

AUX_CYAN = "#8FA6BB"; AUX_CORAL = "#B5838D"; AUX_ORANGE = "#C08B7C"
AUX_PERIW = "#A99AAF"; AUX_SAGE = "#8DA290"; AUX_SAND = "#D9C3A9"
DESIGN_BLUE = "#8FA6BB"

METHOD_COLOR = {
    "PACo-Aptamer": PACO,
    "PACo": PACO,
    "RhoDesign": RHODESIGN,
    "AlignIF": ALIGNIF,
    "NA-MPNN": NAMPNN,
    "gRNAde": GRNADE,
}
METHOD_LABEL = {
    "PACo-Aptamer": "PACo-Aptamer",
    "PACo": "PACo-Aptamer",
    "RhoDesign": "RhoDesign",
    "AlignIF": "AlignIF",
    "NA-MPNN": "NA-MPNN",
    "gRNAde": "gRNAde",
}

METHOD_ORDER = ["PACo-Aptamer", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]

PACO_BLUES = LinearSegmentedColormap.from_list(
    "paco_blues", ["#FFFFFF", "#DCE4EC", "#B9C9D8", "#8FA6BB", "#6B7B8D"])
if "paco_blues" not in mpl.colormaps:
    mpl.colormaps.register(PACO_BLUES, name="paco_blues")

def apply_style():
    mpl.rcParams.update({
        "font.family": "Arial",

        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "mathtext.cal": "Arial:italic",
        "mathtext.default": "regular",
        "font.size": 6.0,
        "axes.titlesize": 6.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.2,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "legend.frameon": False,
        "lines.linewidth": 0.8,
        "pdf.fonttype": 42,
    })

MM = 1 / 25.4

def panel_label(ax, letter, x=None, y=None, size=8):
    """Nature Methods 风格粗体小写 panel 字母。"""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = fig.get_figwidth(), fig.get_figheight()
    dx = (1.2 * MM) / w_in
    dy = (1.2 * MM) / h_in
    fig.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=size,
             fontweight="bold", va="bottom", ha="left", color="black")

def load(rel):
    with open(os.path.join(SUR, rel)) as f:
        return json.load(f)

def save(fig, name, dpi=600, pad=0.02):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=pad)
    print("saved", name)

def pfmt(p):
    if p < 1e-4:
        return f"P = {p:.0e}".replace("e-", " × 10^−")
    if p < 0.001:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"
