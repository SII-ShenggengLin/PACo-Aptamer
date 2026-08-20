# -*- coding: utf-8 -*-
"""Fig 2d | head-to-head 逐链散点:TACo-Aptamer vs RhoDesign(477 链)。

上三角雾蓝底 = TACo 占优;虚线 y = x;链级胜率 + 配对 Wilcoxon Δ/P 面板内标注。
单独运行:python fig2_d.py → out/fig2_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 3:5])
    panel_label(ax, "d", x=-0.21, y=1.1)
    C.draw_h2h(ax, "RhoDesign")
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_d")
