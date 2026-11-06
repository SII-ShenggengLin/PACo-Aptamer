# -*- coding: utf-8 -*-
"""Fig 3c | zero-shot 93 链逐链恢复率优势(PACo-Aptamer − RhoDesign)直方图。

正负分色(与 Fig. 2f 同款),平局为零点处灰色窄条。
单独运行:python fig3_c.py → out/fig3_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axv = fig.add_subplot(gs[0, 4:6])
    C.draw_hist(axv, "zero_shot_targets", "c")
    return axv


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_c")
