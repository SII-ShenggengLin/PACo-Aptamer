# -*- coding: utf-8 -*-
"""Fig 3a | zero-shot 93 链双口径柱状(best/t01),中性样块图例。

单独运行:python fig3_a.py → out/fig3_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 0:2])
    C.draw_bars(axb, "zero_shot_targets", "a")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_a")
