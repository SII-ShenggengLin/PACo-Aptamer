# -*- coding: utf-8 -*-
"""Fig 1c | 两阶段训练:S1 预训练(5,773 train 链,与四基线同集)→ S2 配体条件微调 → ens2 集成。

单独运行:python fig1_c.py → out/fig1_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import TACO, TACO_D, panel_label, save


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:2])
    axc.set_xlim(0, 10); axc.set_ylim(0, 10); axc.axis("off")
    panel_label(axc, "c", x=-0.17, y=1.12)
    C.box(axc, 0.2, 6.3, 4.1, 2.6, C.C_MODEL,
          "S1 pretraining\n5,773 train chains\nsame split as\nbaselines", fs=4.6)
    C.arrow(axc, 2.25, 6.3, 2.25, 5.5)
    C.box(axc, 0.2, 2.8, 4.1, 2.6, C.C_MODEL,
          "S2 finetuning\n+ ligand context\n(complex chains)", fs=4.6)
    C.arrow(axc, 4.3, 4.1, 5.3, 4.1)
    C.box(axc, 5.3, 2.8, 4.5, 2.6, "#DCE6EF",
          "final ensemble\n2 S2 checkpoints\nstepwise prob.\naveraging",
          fs=4.6, ec=TACO, lw=0.8)
    axc.text(2.45, 5.9, "60 epochs", fontsize=4.3, ha="left", color="#545450")
    axc.text(2.25, 2.4, "8 epochs", fontsize=4.3, ha="center", color="#545450")
    axc.text(5.0, 9.2, "two-stage training", fontsize=5.6, ha="center",
             style="italic", color=TACO_D)
    axc.text(7.55, 2.2, "val-selected,\npool size unchanged", fontsize=4.3,
             ha="center", va="center", color="#545450")
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_c")
