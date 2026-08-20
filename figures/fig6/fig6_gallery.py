# -*- coding: utf-8 -*-
"""Fig 6a | 12 案画廊:回折-天然叠加结构网格(NACraft Fig3 式)。

12 条 test 链 × 12 种配体身份(22–119 nt, 9 胜 2 负 1 平):
  每格 = 天然链灰绿半透明 cartoon + TACo-Aptamer 最优设计回折(RhoFold+)紫色
        cartoon 叠加 + 界面残基(<5 Å 配体)雾蓝段 + 配体橙色 sticks
  格上注记 stem·配体名, 格下注记 rec(TACo/RhoDesign)·scTM(TACo/RhoDesign)
图片由 fig6_gallery_pymol.py 预渲染(1200×800 透明底)。
单独运行:python fig6_gallery.py → out/fig6_a.{pdf,png}
"""
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fig6_common as C
from fig_style import MM, ROOT, panel_label, save

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
    ("natv",   "#7FA88B", "native chain"),
    ("desp",   "#9B7EB8", "TACo-Aptamer refold"),
    ("ifblue", "#6C9BD1", "interface residues (<5 \u00c5)"),
    ("ligor",  "#E8A088", "ligand"),
]


def draw(fig):
    W, H = fig.get_figwidth(), fig.get_figheight()
    left, right = 0.010, 0.990
    colw_in = (right - left) * W / 4
    img_w_in = colw_in * 0.965
    img_h_in = img_w_in * 800 / 1200
    row_pitch_in = img_h_in + 0.16
    top_in = H - 0.10

    # 图例条
    lgy = 1 - (0.02 + 0.10) / H
    axl = fig.add_axes([0.02, lgy - 0.012, 0.96, 0.016])
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_axis_off()
    x = 0.008
    for _, col, label in LEGEND:
        axl.add_patch(Rectangle((x, 0.42), 0.018, 0.5, fc=col, ec="none",
                                transform=axl.transAxes))
        axl.text(x + 0.024, 0.62, label, fontsize=5.4, va="center",
                 transform=axl.transAxes)
        x += 0.024 + len(label) * 0.0060 + 0.022
    axl.text(x, 0.62, "in-cell metrics: recovery and scTM as TACo-Aptamer/RhoDesign",
             fontsize=5.0, va="center", style="italic", color="#6E6C68",
             transform=axl.transAxes)

    y_top_in = top_in - 0.13
    for i, (stem, lig, rt, rr, st, sr) in enumerate(CELLS):
        r, c = divmod(i, 4)
        x0 = left + c * (right - left) / 4 + (colw_in - img_w_in) / 2 / W
        y0_in = y_top_in - r * row_pitch_in - img_h_in
        ax = fig.add_axes([x0, y0_in / H, img_w_in / W, img_h_in / H])
        img_path = os.path.join(GALLERY_DIR, f"{stem}.png")
        img = matplotlib.image.imread(img_path)
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        ax.set_title(f"{stem} \u00b7 {lig}", fontsize=5.8, pad=2.6)
        ax.text(0.5, -0.062,
                f"rec {rt:.2f}/{rr:.2f} \u00b7 scTM {st:.2f}/{sr:.2f}",
                ha="center", va="top", fontsize=5.2, color="#4A4844",
                transform=ax.transAxes)
        if i == 0:
            panel_label(ax, "a", x=-0.115, y=1.16, size=8)
    return fig


if __name__ == "__main__":
    fig, _gs = C.make_canvas()
    draw(fig)
    save(fig, "fig6_a")
    print("gallery cells:", len(CELLS))
