"""Extended Data 图公共模块:画布几何 + 数据加载 + 统计标注格式。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load

T93 = load("ligand_novelty.json")
R106 = load("ligand_fingerprint_robustness.json")
LOFO = load("ligand_holdout.json")
FAM95 = load("ligand_families.json")
PRED = {
    "PACo-Aptamer": load("predicted_backbone_paco.json"),
    "RhoDesign": load("predicted_backbone_rhodesign.json"),
    "gRNAde": load("predicted_backbone_grnade.json"),
    "AlignIF": load("predicted_backbone_alignif.json"),
    "NA-MPNN": load("predicted_backbone_nampnn.json"),
}
ZS105 = load("predicted_backbone_zero_shot.json")
SCTM = load("self_consistency.json")
MAIN63 = load("benchmark_cluster.json")
SS97 = load("secondary_structure_robustness.json")
ATT96 = load("ligand_attention.json")

PRED_ORDER = ["PACo-Aptamer", "RhoDesign", "gRNAde", "AlignIF", "NA-MPNN"]
PRED_KEY105 = {"PACo-Aptamer": "PACoAptamer", "RhoDesign": "RhoDesign",
               "gRNAde": "gRNAde", "AlignIF": "AlignIF", "NA-MPNN": "NA-MPNN"}
PRED_LAB = {"PACo-Aptamer": "PACo", "RhoDesign": "RhoDesign", "gRNAde": "gRNAde",
            "AlignIF": "AlignIF", "NA-MPNN": "NA-MPNN"}

def make_canvas(h_mm, nrows, ncols, **gs_kw):
    """183mm 全宽画布;h_mm 高度;返回 (fig, gs)。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(183 * MM, h_mm * MM))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gs_kw)
    return fig, gs

def pfmt_sci(p):
    """P 值格式:≥0.995 → P > 0.99;≥0.05 → P = 0.42;0.01–0.05 → P = 0.022;"""
    if p is None:
        return "P = n.a."
    if p >= 0.995:
        return "P > 0.99"
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

def crystal_recovery(method):
    """Return cluster-test per-chain recovery for the requested method."""
    return MAIN63["per_chain"][PRED_KEY105[method]]

def predbb_recovery(method):
    """五方法 predbb per_chain 统一为 {stem: (best, t01)}。"""
    pc = PRED[method]["per_chain"]
    return {k: (v["recovery_best"], v["recovery_t01_mean"]) for k, v in pc.items()}
