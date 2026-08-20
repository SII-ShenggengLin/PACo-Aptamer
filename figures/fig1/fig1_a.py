# -*- coding: utf-8 -*-
"""Fig 1a | 范式示意:现有逆折叠(靶标不在模型内) vs TACo-Aptamer(配体口袋条件化)。

左白框:结构→序列、配体被虚线挡在模型外;右雾蓝框:backbone + ligand context +
SS pairs 三路输入 → seq design;下方两块对比注释(folding-centric vs binding-centric)。
单独运行:python fig1_a.py → out/fig1_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig1_common as C
from fig_style import LIGAND, TACO, TACO_D, panel_label, save


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0:3])
    axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.axis("off")
    panel_label(axa, "a", x=-0.045, y=1.10)
    # 左半:现有方法
    C.box(axa, 0.1, 5.6, 4.4, 3.3, "white", "", ec="#C4C4C0", lw=0.6)
    axa.text(2.3, 8.55, "Existing inverse folding", ha="center", fontsize=5.6,
             style="italic", color="#545450")
    C.rna_icon(axa, 1.15, 7.3, s=1.0)
    axa.text(1.15, 5.95, "RNA backbone", ha="center", fontsize=4.7,
             color="#545450")
    C.arrow(axa, 1.9, 7.3, 2.75, 7.3)
    C.box(axa, 2.8, 6.5, 1.5, 1.6, C.C_MODEL, "seq\ndesign", fs=4.7)
    axa.text(3.55, 5.95, "sequence", ha="center", fontsize=4.7, color="#545450")
    # 靶标配体(在模型之外,虚线断开)
    C.lig_icon(axa, 3.7, 9.5, s=0.5)
    axa.plot([3.7, 3.7], [8.95, 9.1], color="#B07060", lw=0.8, ls=":")
    axa.text(4.35, 9.5, "target ligand not in model", fontsize=4.6,
             color="#B07060", ha="center", va="center")
    axa.plot([5.15, 5.5], [9.3, 9.7], color="#B07060", lw=1.1)
    axa.plot([5.15, 5.5], [9.7, 9.3], color="#B07060", lw=1.1)
    # 右半:TACo
    C.box(axa, 5.0, 5.6, 4.9, 3.3, "#F3F7FA", "", ec=TACO, lw=0.8)
    axa.text(7.45, 8.55, "TACo-Aptamer (this work)", ha="center", fontsize=5.6,
             style="italic", color=TACO_D)
    C.rna_icon(axa, 5.95, 7.3, s=0.95, color=TACO_D)
    axa.text(5.7, 6.2, "backbone", ha="center", fontsize=4.7, color="#545450")
    C.lig_icon(axa, 6.6, 8.15, s=0.7)
    axa.text(7.5, 6.2, "ligand context", fontsize=4.7,
             color="#B0755F", ha="center", va="center")
    # SS 弧线
    for dy in (0.0, 0.24):
        tt = np.linspace(-0.42, 0.42, 20)
        axa.plot(7.35 + tt, 6.35 + dy + 0.08 * np.cos(tt * 7),
                 color="#7FA88B", lw=0.7)
    axa.text(6.5, 5.62, "SS pairs", ha="center", fontsize=4.7, color="#5A7A62")
    C.arrow(axa, 6.95, 7.25, 7.72, 7.25, color=TACO_D)
    C.box(axa, 7.8, 6.5, 1.6, 1.6, C.C_MODEL, "seq\ndesign", fs=4.7, ec=TACO)
    axa.text(8.75, 5.62, "sequence", ha="center", fontsize=4.7, color="#545450")
    axa.text(4.98, 7.3, "vs", ha="center", va="center", fontsize=5.5,
             color="#8C8C88")
    # 下方对比注释
    C.box(axa, 0.1, 0.4, 4.4, 4.4, C.C_DATA,
          "folding-centric:\nsequence must fold back\nto the given structure",
          fs=4.8, ec="#C4C4C0")
    C.box(axa, 5.0, 0.4, 4.9, 4.4, "#F6E3DC",
          "binding-centric:\nligand pocket atoms condition\nthe design (training-time\ncalibration of interface)",
          fs=4.8, ec="#E8A088", tc="#6B4A3E")
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_a")
