# -*- coding: utf-8 -*-
"""Fig 6a | 8HZD_A 案例结构(PyMOL 渲染嵌入)。

a1 全景:链 A cartoon 灰绿 + 界面残基(<5 Å N618)雾蓝段 + 链 B 淡灰 + N618 stick
a2 口袋放大(<8 Å,同取向,残基编号)
图片由 fig6_pymol.py 预先渲染(透明底 PNG,画幅比例与本面板槽位一致)。
单独运行:python fig6_a.py → out/fig6_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import matplotlib
import matplotlib.image

import fig6_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    ax1 = fig.add_subplot(gs[0, 0:4])
    img = matplotlib.image.imread(C.PYMOL_OVERVIEW)
    ax1.imshow(img, interpolation="none")
    ax1.set_axis_off()
    panel_label(ax1, "b", x=-0.10, y=1.06)
    ax1.set_title("8HZD_A · N618-bound fluorescent aptamer",
                  fontsize=5.4, pad=2.5)

    ax2 = fig.add_subplot(gs[0, 4:7])
    img2 = matplotlib.image.imread(C.PYMOL_POCKET)
    ax2.imshow(img2, interpolation="none")
    ax2.set_axis_off()
    ax2.set_title(f"pocket (< {C.POCKET:.0f} Å)", fontsize=5.4, pad=2.5)
    return ax1, ax2


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig6_b")
    print("iface residues:", C.IFACE)
