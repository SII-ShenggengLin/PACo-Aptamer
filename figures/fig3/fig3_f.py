# -*- coding: utf-8 -*-
"""Fig 3f | independent test 235 链逐链恢复率优势(PACo-Aptamer − RhoDesign)直方图。

与 fig3_c 同布局(共用 fig3_common.draw_hist)。
单独运行:python fig3_f.py → out/fig3_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axv = fig.add_subplot(gs[1, 4:6])
    C.draw_hist(axv, "independent_test", "f")
    return axv


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_f")
