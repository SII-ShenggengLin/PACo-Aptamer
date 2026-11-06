# -*- coding: utf-8 -*-
"""Fig 3b | zero-shot 93 链 head-to-head:PACo-Aptamer vs RhoDesign 逐链散点。

单独运行:python fig3_b.py → out/fig3_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axh = fig.add_subplot(gs[0, 2:4])
    C.draw_h2h(axh, "zero_shot_targets", "b")
    return axh


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_b")
