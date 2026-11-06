# -*- coding: utf-8 -*-
"""Fig 2b | head-to-head 逐链散点:PACo-Aptamer 与 RhoDesign(477 链)。

上三角雾蓝底 = PACo 占优;虚线 y = x;含义由图例说明,面板内不写统计文字。
单独运行:python fig2_b.py → out/fig2_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    C.draw_h2h(ax, "RhoDesign")
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_b")
