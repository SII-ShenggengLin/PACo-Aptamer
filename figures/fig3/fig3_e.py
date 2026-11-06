# -*- coding: utf-8 -*-
"""Fig 3e | independent test 235 链 head-to-head:PACo-Aptamer vs RhoDesign 逐链散点。

与 fig3_b 同布局(共用 fig3_common.draw_h2h)。
单独运行:python fig3_e.py → out/fig3_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axh = fig.add_subplot(gs[1, 2:4])
    C.draw_h2h(axh, "independent_test", "e")
    return axh


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_e")
