# -*- coding: utf-8 -*-
"""Fig 1c | Transformer encoder block 细节(对应 a 中联合注意力模块)。

Pre-LN 结构:LayerNorm → 多头自注意力 → 残差 → LayerNorm → FFN → 残差,
共 3 层(右侧虚线回路);残差支路走左侧正交折线;顶部输出 Updated
tokens。模块框白底黑框;标题带与外框齐边。坐标系与物理轴等比。
"""
from matplotlib.patches import Circle

import fig1_common as C
from fig1_common import fbox, opath, plus, txt

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY


def draw(ax):
    # v1.47 面板收窄(xlim 8.429→7.828),内容整体左移 0.55 重新居中:
    # 内容实际只占 ~6.0 单位(残差干线 1.45 到 ×3 标签 ~7.48),旧版
    # 左右留白 1.21/0.79 失衡(用户反馈左边空白太多),收窄并居中后
    # 两侧各 ~0.66(v1.30 曾以同款"收窄+左移"处理过一次)
    # v1.49 箭头节奏统一:五条竖直箭头一律 0.75(旧 0.90/0.46/0.66/
    # 0.46/1.31,输出箭头独长 1.31 与其他不协调,用户反馈)。各框与
    # ⊕ 在 1.85→7.66 的原总高度内重排,⊕ 移到两侧框的中点(旧版
    # ⊕ 偏上、两侧箭头 0.46/0.66 不对称);输入框与 Updated 框位置
    # 基本不动(7.70→7.66),×3 回路挂点随框中心微调
    C.panel_frame(ax, 0.24, 0.24, 7.35, 9.52, "Transformer encoder ×3",
                  fs=C.FS_HEAD, bh=0.84)
    cx = 3.665

    # ---- 输入 ----
    fbox(ax, 1.165, 1.25, 5.00, 0.60)
    txt(ax, cx, 1.55, "Nucleotide + ligand tokens", fs=SM)

    # ---- LayerNorm + 多头自注意力 → ⊕ ----
    opath(ax, [(cx, 1.85), (cx, 2.60)])
    fbox(ax, 1.615, 2.60, 4.10, 0.80)
    txt(ax, cx, 3.15, "LayerNorm + multi-head", fs=SM)
    txt(ax, cx, 2.85, "self-attention", fs=SM)
    opath(ax, [(cx, 3.40), (cx, 4.15)], ms=4.5)
    plus(ax, cx, 4.29)
    # 残差 1(块输入 → ⊕1)。v1.39 改用 d 图习惯:从主线黑点取流
    # (输入→自注意力箭头段中点),走自己的竖直干线(1.40)上行入 ⊕1;
    # 旧画法从输入框左缘出发、残差 2 又从 ⊕1 右缘折返,两条水平段
    # 同在 y=4.15 且 x 区间重叠,视觉上共线成一根
    ax.add_patch(Circle((cx, 2.225), 0.05, fc="black", ec="none", zorder=6))
    ax.plot([cx, 1.40, 1.40, 3.525], [2.225, 2.225, 4.29, 4.29],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    # ---- LayerNorm + FFN → ⊕ ----
    opath(ax, [(cx, 4.43), (cx, 5.18)])
    fbox(ax, 1.615, 5.18, 4.10, 0.70)
    txt(ax, cx, 5.53, "LayerNorm + FFN", fs=SM)
    opath(ax, [(cx, 5.88), (cx, 6.63)], ms=4.5)
    plus(ax, cx, 6.77)
    # 残差 2(⊕1 后 → ⊕2)。同样从主线黑点取流(⊕1→FFN 箭头段中点),
    # 竖直干线放在残差 1 之外(1.40 → 0.90),与 d 图逐级外移一致
    ax.add_patch(Circle((cx, 4.805), 0.05, fc="black", ec="none", zorder=6))
    ax.plot([cx, 0.90, 0.90, 3.525], [4.805, 4.805, 6.77, 6.77],
            color=C.C_EDGE, lw=0.7, zorder=2, solid_capstyle="round")

    # ---- 输出(顶部,箭头有明确指向) ----
    opath(ax, [(cx, 6.91), (cx, 7.66)])
    fbox(ax, 2.015, 7.66, 3.30, 0.60)
    txt(ax, cx, 7.96, "Updated tokens", fs=SM)

    # ---- ×3 循环(右侧虚线:输出接回第一层输入) ----
    opath(ax, [(5.315, 7.96), (6.465, 7.96), (6.465, 3.00), (5.715, 3.00)],
          ls=(0, (4, 3)), color=C.C_DASH, ms=5.0)
    txt(ax, 6.615, 5.48, "×3", fs=SM, ha="left")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes([C.RECT_BX[1], C.RECT_BCD_Y[0],
                     C.RECT_BCD_WC, C.RECT_BCD_Y[1]])
    ax.set_xlim(C.D_LIM_C[0], C.D_LIM_C[1])
    ax.set_ylim(C.D_LIM_C[2], C.D_LIM_C[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_c")
