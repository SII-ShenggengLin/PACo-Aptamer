# -*- coding: utf-8 -*-
"""Fig 4 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig4_a..i)与组合脚本(fig4_assemble)共用:
- make_canvas():183×159 mm 画布、GridSpec(3,3) —— 与 Fig. 2/3 同款几何,
  九面板严格列/行对齐(2026-10-22 第二轮:应用户要求补入第九面板 i,
  scMCC 碱基对接触保真,自 Supplementary Fig. 10 提升为主图;
  字母 a–h 含义不变,仅物理位置重排为阅读顺序)。
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:消融表 74_p00 / 界面 Wilcoxon 19 / 距离梯度 77_p02_gfix /
  SS 保真 77_p03_gfix / scTM 86_sctm_gfix / scMCC 92_scmcc_gfix
  (gRNAde 列均已 C/G 解码修正)。
- 面板规范与 Fig. 2/3 一致:面板内不写统计文字(数字由正文/图注承载),
  横坐标方法全名(PACo-Aptamer)居中 20°,数字一律不加粗,图例 5.5 pt。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源;与 Fig. 2/3 相同的网格节奏) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 159 * MM
# left 0.105 容 a 的变体名与 g/i 的两行方法名(约 18.5 mm);wspace 0.42 的
# 列间隙约 17.6 mm 容第 2/3 列的 y 轴标签组;hspace 0.40 行间空隙约 15 mm
# (上一行 20° 方法名 + 下一行面板字母)
GRID_KW = dict(hspace=0.40, wspace=0.42,
               left=0.105, right=0.985, top=0.962, bottom=0.062)


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×159mm 画布。

    本图局部放大字号(作者要求,读起来不费劲;不动 fig_style 全局,
    与 fig3 tighten_ylabels 同为按图覆盖):刻度 5.5→6.0、轴标题
    6.0→6.5、图例 5.2→6.0;面板内数值标注 4.8→5.2、g/i 行标签
    4.6→5.0 在各面板脚本中改。
    """
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


# ================= 数据 =================
P00 = load("74_p00_iface_ablation.json")["phase1_ablation"]
IFACE_W = load("19_iface_wilcoxon.json")
P02 = load("77_p02_gfix.json")["methods"]        # gRNAde 列 C/G 解码修正
P03 = load("77_p03_gfix.json")["methods"]        # 同上
SCTM = load("86_sctm_gfix.json")                 # 同上
SCMCC = load("92_scmcc_gfix.json")               # 同上(另含 gRNAde_orig_bugdecode 存档列)

BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]
BIN_LAB = ["≤5", "5–8", "8–12", "12–20", ">20"]

# (展示名, 数据键);正文与横坐标统一用全名 PACo-Aptamer
MKEY = [("PACo-Aptamer", "PACo_ens2"), ("RhoDesign", "RhoDesign"),
        ("AlignIF", "AlignIF"), ("NA-MPNN", "NA-MPNN"), ("gRNAde", "gRNAde")]
