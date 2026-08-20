# -*- coding: utf-8 -*-
"""Fig 3d | zero-shot 93 链逐链 Δ(TACo − RhoDesign)小提琴 + 抖动点。

单独运行:python fig3_d.py → out/fig3_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig3_common as C
from fig_style import save


def draw(fig, gs):
    rng = np.random.default_rng(11)
    axv = fig.add_subplot(gs[0, 8:10])
    C.draw_violin(axv, "zero_shot_targets", "d", rng)
    return axv


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_d")
