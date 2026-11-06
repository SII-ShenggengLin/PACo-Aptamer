# -*- coding: utf-8 -*-
"""Fig 1c | 两阶段训练:S1 预训练(5,773 train 链,与基线同集)→ S2 配体条件微调
→ 验证集选定集成(措辞与图注一致,无图内标题)。

单独运行:python fig1_c.py → out/fig1_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import PACO, panel_label, save


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:2])
    axc.set_xlim(0, 10); axc.set_ylim(0, 10); axc.axis("off")
    panel_label(axc, "c")
    C.box(axc, 0.2, 6.3, 4.5, 2.6, C.C_MODEL,
          "S1 pretraining\n5,773 train chains\nidentical to\n"
          "baseline retraining", fs=4.5)
    C.arrow(axc, 2.25, 6.3, 2.25, 5.5)
    C.box(axc, 0.2, 2.8, 4.5, 2.6, C.C_MODEL,
          "S2 finetuning\n+ ligand context\n(768 complex\nchains)", fs=4.5)
    C.arrow(axc, 4.7, 4.1, 5.15, 4.1)
    C.box(axc, 5.2, 2.8, 4.7, 2.6, C.C_MODEL,
          "final ensemble\nstepwise probability\naveraging of two\n"
          "stage-2 checkpoints", fs=4.4, ec=PACO, lw=0.8)
    axc.text(2.45, 5.9, "60 epochs", fontsize=4.3, ha="left")
    axc.text(2.25, 2.4, "8 epochs", fontsize=4.3, ha="center")
    axc.text(7.55, 2.2, "validation-selected,\npool size unchanged",
             fontsize=4.3, ha="center", va="center")
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_c")
