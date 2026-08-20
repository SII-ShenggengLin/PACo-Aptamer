# -*- coding: utf-8 -*-
"""Fig 4 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig4_a/b/c/f/g)与组合脚本(fig4_assemble)共用:
- make_canvas():同一张 183×128 mm 画布、同一 GridSpec(2,8),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:消融表 74_p00 / 距离梯度 77_p02 / SS 保真 77_p03 / scTM 86。
- panel 字母沿用论文正文约定(a,b,c,f,g;d/e 历史上被裁撤,见 manuscript)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 128 * MM
GRID_KW = dict(hspace=0.55, wspace=1.15, height_ratios=[1.05, 1.15])


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×128mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 8, figure=fig, **GRID_KW)
    return fig, gs


# ================= 数据 =================
P00 = load("74_p00_iface_ablation.json")["phase1_ablation"]
P02 = load("77_p02_distance_gradient.json")["methods"]
P03 = load("77_p03_ss_fidelity.json")["methods"]
SCTM = load("86_sctm.json")

BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]
BIN_LAB = ["≤5", "5–8", "8–12", "12–20", ">20"]
BIN_X = [2.5, 6.5, 10, 16, 25]

MKEY = [("TACo-Aptamer", "TACo_ens2"), ("RhoDesign", "RhoDesign"),
        ("AlignIF", "AlignIF"), ("NA-MPNN", "NA-MPNN"), ("gRNAde", "gRNAde")]
