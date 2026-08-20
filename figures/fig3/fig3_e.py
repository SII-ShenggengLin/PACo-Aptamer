# -*- coding: utf-8 -*-
"""Fig 3e | independent test 235 链(2024+ 时间切分)双口径柱状 + Wilcoxon。

与 fig3_b 同布局(共用 fig3_common.draw_bars)。
单独运行:python fig3_e.py → out/fig3_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import save


def draw(fig, gs):
    axb = fig.add_subplot(gs[1, 3:6])
    C.draw_bars(axb, "independent_test", "A_indep_235", "e",
                "independent test (n = 235, 2024+ temporal split)")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_e")
