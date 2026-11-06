# -*- coding: utf-8 -*-
"""Fig 6a | 12 案画廊:回折-天然叠加结构网格(NACraft Fig3 式)。

12 条 test 链 × 12 种配体身份(22–119 nt, 9 胜 2 负 1 平):
  每格 = 天然链灰绿半透明 cartoon + PACo-Aptamer 最优设计回折(RhoFold+)紫色
        cartoon 叠加 + 界面残基(<5 Å 配体)雾蓝段 + 配体橙色 sticks
  格上注记 stem·配体名, 格下注记 rec(PACo/RhoDesign)·scTM(PACo/RhoDesign)
  图例在画廊下方水平居中(2026-10-23 自上方移下;每格指标口径 PACo/RhoDesign
  由图题说明,不再在图内重复)
图片由 fig6_gallery_pymol.py 预渲染(1200×800 透明底)。
单独运行:python fig6_gallery.py → out/fig6_a.{pdf,png}
"""
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fig6_common as C
from fig_style import ROOT, panel_label, save

GALLERY_DIR = os.path.join(ROOT, "figures", "out", "fig6_gallery")

# stem, 配体显示名, rec_T, rec_R, scTM_T, scTM_R(池内 best 口径, 86 号 scTM)
CELLS = [
    ("3GAO_A", "xanthine",         0.70, 0.61, 0.54, 0.12),
    ("6UC7_B", "N2-acetylguanine", 0.72, 0.64, 0.71, 0.31),
    ("7Q80_E", "guanine",          0.88, 0.67, 0.24, 0.23),
    ("3FU2_C", "preQ1",            0.78, 0.66, 0.27, 0.15),
    ("8D5L_A", "TAL1",             0.94, 0.79, 0.44, 0.27),
    ("6UET_A", "SAM",              0.69, 0.65, 0.10, 0.10),
    ("5BTP_B", "ZMP",              0.76, 0.69, 0.19, 0.09),
    ("8HB1_A", "NMN",              0.78, 0.70, 0.14, 0.12),
    ("4K31_B", "apramycin",        1.00, 0.95, 0.22, 0.21),
    ("4FE5_B", "hypoxanthine",     0.70, 0.63, 0.58, 0.40),
    ("6PRV_D", "GTP",              0.72, 0.74, 0.40, 0.10),
    ("6E8S_B", "TO1-biotin",       0.68, 0.74, 0.28, 0.33),
]

LEGEND = [
    ("natv",   "#A3B18A", "native chain"),
    ("desp",   "#A99AAF", "PACo-Aptamer refold"),
    ("ifblue", "#6B7B8D", "interface residues (<5 \u00c5)"),
    ("ligor",  "#C08B7C", "ligand"),
]

LEGEND_FS = 8.0          # 与全图 8.0 pt 图例字号一致(第五轮 +1 pt)
_SWATCH, _GAP, _SEP = 0.022, 0.008, 0.030   # fig fraction


def _draw_legend(fig):
    """画廊下方水平居中的色块图例条(按实测文本宽度居中)。"""
    y = 1 - (C.GAL_BOTTOM_IN + 0.18) / fig.get_figheight()
    r = fig.canvas.get_renderer()
    texts = [fig.text(0.5, y, lab, fontsize=LEGEND_FS, va="center",
                      ha="left", color="black") for _, _, lab in LEGEND]
    fig.canvas.draw()
    widths = [(t.get_window_extent(r).width / fig.get_figwidth()
               / fig.dpi) for t in texts]
    for t in texts:
        t.remove()
    total = (sum(_SWATCH + _GAP + w for w in widths)
             + _SEP * (len(LEGEND) - 1))
    x = 0.5 * (1 - total)
    for (_, col, label), w in zip(LEGEND, widths):
        fig.patches.append(Rectangle((x, y - 0.0075), _SWATCH, 0.012,
                                     transform=fig.transFigure, fc=col,
                                     ec="none", zorder=5))
        fig.text(x + _SWATCH + _GAP, y, label, fontsize=LEGEND_FS,
                 va="center", ha="left", color="black")
        x += _SWATCH + _GAP + w + _SEP


def draw(fig):
    W, H = fig.get_figwidth(), fig.get_figheight()
    img_w_in = C.GAL_IMG_W_IN
    img_h_in = C.GAL_IMG_H_IN
    row_pitch_in = C.GAL_PITCH_IN
    y_top_in = H - C.GAL_TOP_IN

    for i, (stem, lig, rt, rr, st, sr) in enumerate(CELLS):
        r, c = divmod(i, 4)
        x0 = C.IMG_X[c] / W
        y0_in = y_top_in - r * row_pitch_in - img_h_in
        ax = fig.add_axes([x0, y0_in / H, img_w_in / W, img_h_in / H])
        img_path = os.path.join(GALLERY_DIR, f"{stem}.png")
        img = matplotlib.image.imread(img_path)
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        # 2026-10-23:格题/注记与结构图净空 3.5/5 pt;第五轮格题 8.5 pt、
        # 注记 7.5 pt(注记 8 pt 时超出列宽 ~2 pt,降半点后净空 ~2 mm)
        ax.set_title(f"{stem} \u00b7 {lig}", fontsize=8.5, pad=3.5)
        ax.annotate(f"recovery {rt:.2f}/{rr:.2f} \u00b7 scTM {st:.2f}/{sr:.2f}",
                    xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5)
        if i == 0:
            panel_label(ax, "a", size=10)
    _draw_legend(fig)
    return fig


if __name__ == "__main__":
    fig, _L = C.make_canvas()
    draw(fig)
    save(fig, "fig6_a")
    print("gallery cells:", len(CELLS))
