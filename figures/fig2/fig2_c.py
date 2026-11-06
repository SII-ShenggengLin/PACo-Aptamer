# -*- coding: utf-8 -*-
"""Fig 2c | head-to-head 逐链散点:PACo-Aptamer 与 AlignIF(477 链)。

布局与 fig2_b 相同(共用 fig2_common.draw_h2h)。
单独运行:python fig2_c.py → out/fig2_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    C.draw_h2h(ax, "AlignIF")
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_c")
