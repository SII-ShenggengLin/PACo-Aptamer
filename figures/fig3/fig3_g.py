# -*- coding: utf-8 -*-
"""Fig 3g | independent test 235 链逐链 Δ(TACo − RhoDesign)小提琴 + 抖动点。

与 fig3_d 同布局(共用 fig3_common.draw_violin)。
单独运行:python fig3_g.py → out/fig3_g.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig3_common as C
from fig_style import save


def draw(fig, gs):
    rng = np.random.default_rng(11)
    axv = fig.add_subplot(gs[1, 8:10])
    C.draw_violin(axv, "independent_test", "g", rng)
    return axv


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_g")
