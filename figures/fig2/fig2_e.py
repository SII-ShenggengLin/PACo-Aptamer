# -*- coding: utf-8 -*-
"""Fig 2e | head-to-head 逐链散点:TACo-Aptamer vs AlignIF(477 链)。

布局与 fig2_d 相同(共用 fig2_common.draw_h2h)。
单独运行:python fig2_e.py → out/fig2_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 5:8])
    panel_label(ax, "e", x=-0.21, y=1.1)
    C.draw_h2h(ax, "AlignIF")
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_e")
