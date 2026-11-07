"""Fig 4 公共模块:统一画布几何 + 数据准备。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load

FIG_W_IN = 183 * MM
FIG_H_IN = 159 * MM

GRID_KW = dict(hspace=0.40, wspace=0.42,
               left=0.105, right=0.985, top=0.962, bottom=0.062)

def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×159mm 画布。"""
    apply_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    mpl.rcParams.update({
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0, "axes.labelsize": 6.5,
    })
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(3, 3, figure=fig, **GRID_KW)
    return fig, gs

def tighten_ylabels(fig):
    """列间呼吸感:y 刻度签 pad 3.5→1.5 pt、y 轴标签 labelpad 4→1.5 pt。"""
    for ax in fig.axes:
        ax.tick_params(axis="y", pad=1.5)
        lbl = ax.yaxis.label.get_text()
        if lbl:
            ax.set_ylabel(lbl, labelpad=1.5)

P00 = load("ablation.json")["phase1_ablation"]
IFACE_W = load("interface_statistics.json")
P02 = load("interface_distance.json")["methods"]
P03 = load("secondary_structure_fidelity.json")["methods"]
SCTM = load("self_consistency.json")
SCMCC = load("secondary_structure_contact_recovery.json")

BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]
BIN_LAB = ["≤5", "5–8", "8–12", "12–20", ">20"]

MKEY = [("PACo-Aptamer", "PACo_ens2"), ("RhoDesign", "RhoDesign"),
        ("AlignIF", "AlignIF"), ("NA-MPNN", "NA-MPNN"), ("gRNAde", "gRNAde")]
