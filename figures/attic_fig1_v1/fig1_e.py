# -*- coding: utf-8 -*-
"""Fig 1e | Transformer decoder block 细节(对应总览 a 中同名模块)。

已生成前缀 → 掩码自注意力(下三角因果掩码)→ 残差 → 交叉注意力
(K/V 来自编码器的核苷酸+配体 token,黑色虚线)→ 残差 → FFN → 残差
→ softmax 采样下一个核苷酸;采样结果经左侧正交回路送回前缀
(autoregressive)。共 3 层。所有线框/箭头/文字黑色。坐标系等比。
单独运行:python fig1_e.py → out/fig1_e.{pdf,png}
"""
from matplotlib.patches import Circle, Rectangle

import fig1_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 3])
    ax.set_xlim(0, C.D_LIM[0]); ax.set_ylim(0, C.D_LIM[1])
    ax.set_aspect("equal"); ax.set_axis_off()
    panel_label(ax, "e")

    C.lab(ax, 5.0, 10.35, "Transformer decoder ×3", fs=C.FS_HEAD)

    # ---- 输入:已生成前缀(A C G + 虚线下一格) ----
    C.seq_strip(ax, 4.42, 0.95, letters="ACG", cw=0.62, h=0.50)
    C.lab(ax, 7.05, 1.20, "Generated prefix", fs=C.FS_SM, ha="left")
    C.opath(ax, [(5.6, 1.45), (5.6, 2.02)], lw=0.9, ms=5.5)
    ax.add_patch(Circle((5.6, 1.75), 0.055, fc="black", ec="none", zorder=6))

    # ---- 掩码自注意力 → ⊕ ----
    C.wbox(ax, 3.5, 2.02, 4.2, 1.58, lw=0.9, style="round,pad=0.04")
    C.lab(ax, 5.6, 3.26, "Masked self-attention", fs=C.FS_TINY)
    C.causal_glyph(ax, 5.17, 2.16, 0.17, n=5)
    C.opath(ax, [(5.6, 3.60), (5.6, 3.68)], lw=0.9, ms=4.5)
    C.plus(ax, 5.6, 3.92)
    C.opath(ax, [(5.6, 1.75), (2.6, 1.75), (2.6, 3.92), (5.34, 3.92)],
            lw=0.8)

    # ---- 交叉注意力(K/V 来自编码器)→ ⊕ ----
    C.opath(ax, [(5.6, 4.14), (5.6, 4.50)], lw=0.9, ms=5.5)
    ax.add_patch(Circle((5.6, 4.31), 0.055, fc="black", ec="none", zorder=6))
    C.wbox(ax, 3.9, 4.50, 3.4, 1.25, lw=0.9, style="round,pad=0.04")
    C.lab(ax, 5.6, 5.12, "Cross-attention", fs=C.FS_SM)
    C.token_strip(ax, 7.75, 4.87, n_rna=3, n_lig=2, s=0.24)
    C.opath(ax, [(7.70, 5.0), (7.35, 5.0)], lw=0.9, ls=(0, (4, 3)), ms=5.5)
    C.lab(ax, 8.5, 6.06, "Encoder tokens", fs=C.FS_TINY)
    C.lab(ax, 8.5, 5.72, "(K, V)", fs=C.FS_TINY)
    C.opath(ax, [(5.6, 5.75), (5.6, 5.88)], lw=0.9, ms=4.5)
    C.plus(ax, 5.6, 6.12)
    C.opath(ax, [(5.6, 4.31), (3.3, 4.31), (3.3, 6.12), (5.34, 6.12)],
            lw=0.8)

    # ---- FFN → ⊕ ----
    C.opath(ax, [(5.6, 6.34), (5.6, 6.70)], lw=0.9, ms=5.5)
    ax.add_patch(Circle((5.6, 6.51), 0.055, fc="black", ec="none", zorder=6))
    C.wbox(ax, 4.4, 6.70, 2.4, 0.80, lw=0.9, style="round,pad=0.04")
    C.lab(ax, 5.6, 7.10, "FFN", fs=C.FS_TXT)
    C.opath(ax, [(5.6, 7.50), (5.6, 7.63)], lw=0.9, ms=4.5)
    C.plus(ax, 5.6, 7.87)
    C.opath(ax, [(5.6, 6.51), (4.0, 6.51), (4.0, 7.87), (5.34, 7.87)],
            lw=0.8)

    # ---- softmax → 采样(字母在条柱下缘,箭头止于字母行下方) ----
    C.opath(ax, [(5.6, 8.09), (5.6, 8.26)], lw=0.9, ms=4.5)
    for k, (h, ch) in enumerate(zip((0.60, 0.27, 0.42, 0.16), "ACGU")):
        x0 = 4.85 + k * 0.50
        ax.add_patch(Rectangle((x0, 8.62), 0.34, h, fc=C.C_RNA_L,
                               ec="black", lw=0.9, zorder=3))
        ax.text(x0 + 0.17, 8.36, ch, fontsize=C.FS_TINY, ha="center",
                va="center", color="black")
    C.lab(ax, 5.6, 9.44, "P(next)", fs=C.FS_TINY)
    ax.add_patch(Rectangle((7.3, 8.65), 0.62, 0.55, fc="white", ec="black",
                           lw=1.0, zorder=3))
    C.lab(ax, 7.61, 8.925, "U", fs=C.FS_TXT, zorder=4)
    C.opath(ax, [(6.66, 8.925), (7.28, 8.925)], lw=0.9, ms=5.5)
    C.lab(ax, 7.9, 9.42, "Sample", fs=C.FS_TINY, ha="left")

    # ---- 自回归回路(左侧正交折线,从样本格顶部送回前缀虚线格底部) ----
    C.opath(ax, [(7.61, 9.20), (7.61, 9.75), (1.1, 9.75), (1.1, 0.60),
                 (6.55, 0.60), (6.55, 0.93)], lw=0.9)
    C.lab(ax, 0.72, 5.5, "Autoregressive", fs=C.FS_SM, rot=90)
    return ax


if __name__ == "__main__":
    import fig1_common as C
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_e")
