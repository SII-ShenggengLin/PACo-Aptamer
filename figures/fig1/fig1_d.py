# -*- coding: utf-8 -*-
"""Fig 1d | Transformer decoder block 细节(对应 a 中自回归解码模块)。

Pre-LN 结构:LayerNorm+masked self-attention → 残差 → LayerNorm+
cross-attention → 残差 → LayerNorm+FFN → 残差 → softmax → 采样下一个
核苷酸,采样结果经右侧正交回路送回前缀(autoregressive)。三个子层
框等宽;模块框白底黑框;标题带与外框齐边。坐标系与物理轴等比。
"""
from matplotlib.patches import Circle, Rectangle

import fig1_common as C
from fig1_common import fbox, opath, plus, txt

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY


def draw(ax):
    # v1.48 面板收窄(xlim 9.597→8.884,行宽让给 b)并内容整体左移
    # 0.30 重新居中:v1.47 加宽后两侧留白 ~0.7/0.6 偏大(用户反馈),
    # 收窄居中后两侧各 ~0.45
    C.panel_frame(ax, 0.24, 0.24, 8.40, 9.52, "Transformer decoder ×3",
                  fs=C.FS_HEAD, bh=0.84)
    cx = 4.51

    # ---- 输入:已生成前缀(A U G + 虚线下一格) ----
    for k, ch in enumerate("AUG"):
        x0 = 3.30 + k * 0.62
        ax.add_patch(Rectangle((x0, 0.85), 0.56, 0.50, fc=C.NT_COLORS[ch],
                               ec="white", lw=0.5, zorder=3))
        txt(ax, x0 + 0.28, 1.10, ch, fs=SM, color="white", weight="bold",
            z=4)
    fbox(ax, 5.16, 0.85, 0.56, 0.50, ls=(0, (3, 2)), r=0.02)
    txt(ax, cx, 0.52, "Partial sequence", fs=SM)

    # ---- LayerNorm + masked self-attention → ⊕ ----
    opath(ax, [(cx, 1.35), (cx, 1.95)], ms=4.5)
    ax.add_patch(Circle((cx, 1.65), 0.05, fc="black", ec="none", zorder=6))
    fbox(ax, 2.01, 1.95, 5.00, 0.80)
    txt(ax, cx, 2.50, "LayerNorm + masked", fs=SM)
    txt(ax, cx, 2.20, "self-attention", fs=SM)
    opath(ax, [(cx, 2.75), (cx, 3.21)], ms=4.5)
    plus(ax, cx, 3.35)
    # 残差 1(块输入 → ⊕1)
    ax.plot([cx, 1.65, 1.65, 4.37], [1.65, 1.65, 3.35, 3.35],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    # ---- LayerNorm + cross-attention → ⊕ ----
    opath(ax, [(cx, 3.49), (cx, 4.05)], ms=4.5)
    ax.add_patch(Circle((cx, 3.77), 0.05, fc="black", ec="none", zorder=6))
    fbox(ax, 2.01, 4.05, 5.00, 0.70)
    txt(ax, cx, 4.40, "LayerNorm + cross-attention", fs=SM)
    opath(ax, [(cx, 4.75), (cx, 5.21)], ms=4.5)
    plus(ax, cx, 5.35)
    # 残差 2(⊕1 后 → ⊕2)
    ax.plot([cx, 1.20, 1.20, 4.37], [3.77, 3.77, 5.35, 5.35],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    # ---- LayerNorm + FFN → ⊕ ----
    opath(ax, [(cx, 5.49), (cx, 6.05)], ms=4.5)
    ax.add_patch(Circle((cx, 5.77), 0.05, fc="black", ec="none", zorder=6))
    fbox(ax, 2.01, 6.05, 5.00, 0.70)
    txt(ax, cx, 6.40, "LayerNorm + FFN", fs=SM)
    opath(ax, [(cx, 6.75), (cx, 7.16)], ms=4.5)
    plus(ax, cx, 7.30)
    # 残差 3(⊕2 后 → ⊕3)
    ax.plot([cx, 0.75, 0.75, 4.37], [5.77, 5.77, 7.30, 7.30],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    # ---- softmax → 采样 ----
    opath(ax, [(cx, 7.44), (cx, 8.00)], ms=4.5)
    for k, (h, ch) in enumerate(zip((0.34, 0.22, 0.48, 0.13), "ACGU")):
        x0 = 3.59 + k * 0.50
        ax.add_patch(Rectangle((x0, 8.00), 0.34, h, fc=C.NT_COLORS[ch],
                               ec="black", lw=0.5, zorder=3))
        txt(ax, x0 + 0.17, 7.72, ch, fs=TINY)
    txt(ax, cx, 8.66, "P(next)", fs=TINY)
    fbox(ax, 5.90, 7.99, 0.56, 0.50, r=0.02)
    txt(ax, 6.18, 8.24, "G", fs=SM)
    opath(ax, [(5.45, 8.24), (5.90, 8.24)], ms=4.5)
    txt(ax, 6.18, 7.75, "Sample", fs=TINY)

    # ---- 自回归回路(右侧正交折线:样本 → 前缀虚线格) ----
    # v1.35 顶部横线 8.88→8.74(P(next) 8.74→8.66):标题带加高后
    # 与带底(8.92)保持净空
    opath(ax, [(6.18, 8.49), (6.18, 8.74), (7.45, 8.74), (7.45, 1.10),
               (5.72, 1.10)], ms=4.5)
    # 旋转标签在回路线(7.45)与面板框右边(8.64)之间居中(v1.48 随
    # 内容左移与框收窄,8.55→8.05)
    txt(ax, 8.05, 4.75, "Autoregressive", fs=SM, rotation=90)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes([C.RECT_BX[2], C.RECT_BCD_Y[0],
                     C.RECT_BCD_WD, C.RECT_BCD_Y[1]])
    ax.set_xlim(C.D_LIM[0], C.D_LIM[1])
    ax.set_ylim(C.D_LIM[2], C.D_LIM[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_d")
