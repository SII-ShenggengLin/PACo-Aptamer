# -*- coding: utf-8 -*-
"""Fig 3f | independent test 235 链 head-to-head:TACo vs RhoDesign 逐链散点。

与 fig3_c 同布局(共用 fig3_common.draw_h2h)。
单独运行:python fig3_f.py → out/fig3_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axh = fig.add_subplot(gs[1, 6:8])
    C.draw_h2h(axh, "independent_test", "f")
    return axh


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_f")
