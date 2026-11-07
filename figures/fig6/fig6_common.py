"""Shared canvas geometry and final data for Figure 6."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import FIGURE_ASSETS, MM, apply_style, load

FIG_W_IN = 183 * MM
GL, GR = 0.010, 0.990
COLS = [GL + i * (GR - GL) / 4 for i in range(5)]
CELL_IN = (GR - GL) * FIG_W_IN / 4
GAL_IMG_FRAC = 0.72

GAL_IMG_W_IN = CELL_IN * GAL_IMG_FRAC
GAL_INSET = (CELL_IN - GAL_IMG_W_IN) / 2
IMG_X = [c * FIG_W_IN + GAL_INSET for c in COLS[:-1]]
IMG_R = [c * FIG_W_IN - GAL_INSET for c in COLS[1:]]
GAL_IMG_H_IN = GAL_IMG_W_IN * 800 / 1200
GAL_PITCH_IN = GAL_IMG_H_IN + 0.56

GAL_TOP_IN = 0.32
GAL_CAP_IN = 0.24
GAL_BOTTOM_IN = GAL_TOP_IN + 2 * GAL_PITCH_IN + GAL_IMG_H_IN + GAL_CAP_IN
GAL_LEGEND_IN = 0.43

ROW2_TOP_IN = GAL_BOTTOM_IN + GAL_LEGEND_IN + 0.12
B_IMG_FRAC = 0.92

B_IMG_W_IN = CELL_IN * B_IMG_FRAC
ROW2_H_IN = 1.42

ROW3_TOP_IN = ROW2_TOP_IN + ROW2_H_IN + 0.34
ROW3_H_IN = 1.70
FIG_H_IN = ROW3_TOP_IN + ROW3_H_IN + 0.21

D_X0_IN = IMG_X[0] + 0.17
D_X1_IN = COLS[2] * FIG_W_IN - 0.42
E_X0_IN = COLS[2] * FIG_W_IN - 0.28
F_X0_IN = COLS[3] * FIG_W_IN - 0.055
F_GAP_IN = 0.56

E_X1_IN = F_X0_IN - F_GAP_IN
F_X1_IN = COLS[4] * FIG_W_IN + 0.028

def _rect(x0_in, y_top_in, w_in, h_in):
    """左上角 + 宽高(英寸,自页顶量)→ fig-fraction 矩形。"""
    return [x0_in / FIG_W_IN, (FIG_H_IN - y_top_in - h_in) / FIG_H_IN,
            w_in / FIG_W_IN, h_in / FIG_H_IN]

def make_canvas():
    """返回 (fig, L):L 给出 b/c/d/e/f 各面板的 axes 矩形(fig fraction)。"""
    apply_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "axes.labelsize": 8.5,
    })
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    L = {

        "b1": _rect(IMG_X[0], ROW2_TOP_IN, GAL_IMG_W_IN, ROW2_H_IN),
        "b2": _rect(IMG_X[1], ROW2_TOP_IN, GAL_IMG_W_IN, ROW2_H_IN),
        "c":  _rect(COLS[2] * FIG_W_IN + 0.015, ROW2_TOP_IN,
                    COLS[4] * FIG_W_IN - COLS[2] * FIG_W_IN - 0.015, ROW2_H_IN),
        "d":  _rect(D_X0_IN, ROW3_TOP_IN, D_X1_IN - D_X0_IN, ROW3_H_IN),
        "e":  _rect(E_X0_IN, ROW3_TOP_IN, E_X1_IN - E_X0_IN, ROW3_H_IN),
        "f":  _rect(F_X0_IN, ROW3_TOP_IN, F_X1_IN - F_X0_IN, ROW3_H_IN),
    }
    return fig, L

def panel_label_at(fig, letter, x_in, y_top_in, size=10):
    """按画廊网格的固定 x(英寸,自左)放面板字母,与 fig_style.panel_label"""
    dx = (1.2 * MM) / FIG_W_IN
    dy = (1.2 * MM) / FIG_H_IN
    fig.text(x_in / FIG_W_IN - dx, 1 - y_top_in / FIG_H_IN + dy, letter,
             fontsize=size, fontweight="bold", va="bottom", ha="left",
             color="black")

PYMOL_OVERVIEW = os.path.join(FIGURE_ASSETS, "fig6", "fig6_pymol_overview.png")
PYMOL_POCKET = os.path.join(FIGURE_ASSETS, "fig6", "fig6_pymol_pocket.png")

CASE_STUDY = load("case_study.json")
natA = CASE_STUDY["native"]
RESIDS = CASE_STUDY["residue_ids"]
IFACE = CASE_STUDY["interface_residue_ids"]
SEQS = {
    method: (record["sequence"], record["recovery"])
    for method, record in CASE_STUDY["designs"].items()
}

RETROSPECTIVE = load("retrospective_validation.json")
D_APT = np.asarray(RETROSPECTIVE["aptadb_logp_delta"])
D_UTX = np.asarray(RETROSPECTIVE["utexas_logp_delta"])
FAM_RHO = np.asarray(RETROSPECTIVE["family_spearman"])
FAM_N = RETROSPECTIVE["family_sizes"]
D_ALL = np.concatenate([D_APT, D_UTX])

CS = load("screening_enrichment.json")["curves"]

CASE = ["8HZD_A", "8HZD_B", "8HZE_A"]
