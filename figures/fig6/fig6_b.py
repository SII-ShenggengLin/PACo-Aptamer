# -*- coding: utf-8 -*-
"""Fig 6b | 8HZD_A 案例结构(PyMOL 渲染嵌入 + 格题/注记)。

左 全景:链 A cartoon 灰绿 + 界面残基(<5 Å N618)雾蓝段 + 链 B 淡灰 + N618 stick
右 口袋放大(<8 Å,同取向)
两幅渲染列心对齐画廊(panel a)前两列(2026-10-23 第三轮起显示宽 = 列宽的
0.75,自画廊图像宽 0.62 放大,面板不再显扁);每幅渲染上下各一行文字
(与画廊格的 格题/注记 排版一致):
  左  上 8HZD_A · N618            下 7 interface residues (<5 Å)
  右  上 pocket close-up (<8 Å)    下 interface residues 7/8/14/15/28/29/30
渲染图内不嵌字(旧 N618 标签/残基号在打印宽下压叠不可读)。
axes 精确包裹渲染图(高按 PNG 实际宽高比),标题 pad/注记偏移与画廊一致;
整格(格题 + 渲染 + 注记)在 b/c 行内垂直居中。编号 b 仍锚定画廊列 1 图像
左缘(panel_label_at),与编号 a/d 成列对齐。
单独运行:python fig6_b.py → out/fig6_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import matplotlib
import matplotlib.image

import fig6_common as C
from fig_style import save

TITLE_FS, CAP_FS = 8.5, 7.5          # 与画廊格题 8.5 / 注记 7.5 一致(第五轮)
TITLE_PAD, CAP_OFFSET = 3.5, 5.0     # pt,同 fig6_gallery
TITLE_BAND_IN = 0.187                # 格题带高(pad 3.5 pt + 8.5 pt 行高)
CAP_BAND_IN = 0.173                  # 注记带高(偏移 5 pt + 7.5 pt 行高)


def _cell_rect(png_path, col):
    """第 col 列渲染图的 axes 矩形(fig fraction):宽 = 列宽 × 0.75、列心
    对齐,高按 PNG 实际宽高比;整格在 b/c 行内垂直居中。"""
    img = matplotlib.image.imread(png_path)
    w_in = C.B_IMG_W_IN
    h_in = w_in * img.shape[0] / img.shape[1]
    x0_in = C.COLS[col] * C.FIG_W_IN + (C.CELL_IN - w_in) / 2
    slack = max(0.0, C.ROW2_H_IN - TITLE_BAND_IN - h_in - CAP_BAND_IN)
    top_in = C.ROW2_TOP_IN + slack / 2 + TITLE_BAND_IN
    return [x0_in / C.FIG_W_IN,
            (C.FIG_H_IN - top_in - h_in) / C.FIG_H_IN,
            w_in / C.FIG_W_IN, h_in / C.FIG_H_IN]


def _draw_cell(fig, Ldict, key, png_path, col, title, caption):
    rect = _cell_rect(png_path, col)
    Ldict[key] = rect                      # 覆盖占位矩形,供审计/复用
    ax = fig.add_axes(rect)
    img = matplotlib.image.imread(png_path)
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    ax.set_title(title, fontsize=TITLE_FS, pad=TITLE_PAD)
    ax.annotate(caption, xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -CAP_OFFSET), textcoords="offset points",
                ha="center", va="top", fontsize=CAP_FS)
    return ax


def draw(fig, L):
    ax1 = _draw_cell(fig, L, "b1", C.PYMOL_OVERVIEW, 0,
                     "8HZD_A \u00b7 N618", "7 interface residues (<5 \u00c5)")
    C.panel_label_at(fig, "b", C.IMG_X[0], C.ROW2_TOP_IN, size=10)
    _draw_cell(fig, L, "b2", C.PYMOL_POCKET, 1,
               "pocket close-up (<8 \u00c5)",
               "interface residues 7/8/14/15/28/29/30")
    return ax1


if __name__ == "__main__":
    fig, _L = C.make_canvas()
    draw(fig, _L)
    save(fig, "fig6_b")
    print("iface residues:", C.IFACE)
