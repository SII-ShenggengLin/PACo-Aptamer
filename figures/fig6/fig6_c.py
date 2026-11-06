# -*- coding: utf-8 -*-
"""Fig 6c | 逐位恢复序列条(8HZD_A:native + 5 方法最优设计)。

行 = native + 各方法 best sample;列 = 序列位置;填色 = 与 native 一致;
右侧两列 = 整体恢复率 % 与界面位(<5 Å N618,残基 7/8/14/15/28/29/30)恢复率 %,
口径写进图题。面板跨画廊(panel a)第 3–4 列图像,左右缘与其对齐。
(2026-10-23 第二轮:b/c 行加高至 0.95 in,行距放宽,字母 5.5→6.0 pt、
行名/数值 5.8→6.5 pt。)
单独运行:python fig6_c.py → out/fig6_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
import matplotlib.pyplot as plt

import fig6_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save

FS_LETTER, FS_LABEL, FS_VAL = 7.0, 7.5, 7.5   # 第五轮可读性放大(+1 pt)
FS_HEADER = 7.0                          # 列头(与刻度签同号;两列拉开防相碰)
COL_ID, COL_IF = 39.4, 47.4              # 数值列 x(自 37.6/42.8 拉开,容纳
                                          # 7 pt 列头 identity/interface)


def draw(fig, L):
    axb = fig.add_axes(L["c"])
    panel_label(axb, "c", size=10)
    rows = [("native", C.natA, None, None)] + [
        (m, C.SEQS[m][0], C.SEQS[m][1], None) for m in METHOD_ORDER]
    L = len(C.natA)
    iface_pos = [C.RESIDS.index(rn) for rn in C.IFACE]   # 0 基序列位置

    def iface_id(seq):
        return np.mean([seq[p] == C.natA[p] for p in iface_pos])

    for ri, (name, sq, bv, _) in enumerate(rows):
        y0 = len(rows) - 1 - ri        # native 顶行
        if name == "native":
            for pi, ch_ in enumerate(sq):
                axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                         fontsize=FS_LETTER, color="black")
            axb.text(-0.7, y0 + 0.5, "native", ha="right", va="center",
                     fontsize=FS_LABEL, color="black")
            continue
        col = METHOD_COLOR[name]
        for pi, ch_ in enumerate(sq):
            match = ch_ == C.natA[pi]
            axb.add_patch(plt.Rectangle((pi, y0), 1, 1,
                                        fc=col if match else "white",
                                        ec="#D3B5B0", lw=0.3))
            axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                     fontsize=FS_LETTER, color="black")
        axb.text(-0.7, y0 + 0.5, name, ha="right", va="center",
                 fontsize=FS_LABEL, color="black")
        fi = iface_id(sq)
        axb.text(COL_ID, y0 + 0.5, f"{100*bv:.0f}%", ha="center", va="center",
                 fontsize=FS_VAL, color="black")
        axb.text(COL_IF, y0 + 0.5, f"{100*fi:.0f}%", ha="center", va="center",
                 fontsize=FS_VAL, color="black")
    axb.text(COL_ID, 6.45, "identity", ha="center", va="bottom",
             fontsize=FS_HEADER, color="black")
    axb.text(COL_IF, 6.45, "interface", ha="center", va="bottom",
             fontsize=FS_HEADER, color="black")
    axb.set_xlim(-13.5, 52.4)   # 左槽容纳 7.5 pt 行名且不与放大后的 b2 渲染
                                 # 相碰(第六轮);右侧随数值列拉开加宽
    axb.set_ylim(-0.45, 7.35)
    axb.axis("off")
    return axb


if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_c")
    for m in METHOD_ORDER:
        s, v = C.SEQS[m]
        ii = np.mean([s[p] == C.natA[p] for p in
                      [C.RESIDS.index(rn) for rn in C.IFACE]])
        print(f"{m:14s} overall {100*v:5.1f}%  iface {100*ii:5.1f}%")
