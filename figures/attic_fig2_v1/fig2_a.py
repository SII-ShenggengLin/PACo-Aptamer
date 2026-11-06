# -*- coding: utf-8 -*-
"""Fig 2a | 评估协议流程:条件输入 → 池采样 → 双口径恢复 → 统计检验。

单独运行:python fig2_a.py → out/fig2_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_DATA, C_MODEL, C_MET = "#D9C3A9", "#8FA6BB", "#CDBF9E"


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0:3]); axa.set_xlim(0, 10); axa.set_ylim(0, 6.4)
    axa.axis("off")
    panel_label(axa, "a")
    boxes = [
        (0.1, 4.0, 3.3, 1.85, C_DATA,
         "backbone +\npredicted SS\n(same for all)"),
        (3.75, 4.0, 3.0, 1.85, C_MODEL,
         "100 sequences ×\n3 temperatures\nper method"),
        (7.15, 4.0, 2.75, 1.85, C_MET,
         "recovery: best\nsample / t = 0.1\n+ interface (<5 Å)"),
        (0.1, 0.7, 3.3, 1.85, C_MODEL,
         "same-data\nretraining\n(5 methods)"),
        (3.75, 0.7, 6.15, 1.85, C_MET,
         "per-chain paired Wilcoxon\n+ bootstrap 95% CI\nagainst strongest baseline"),
    ]
    for x, y, w, h, fc, txt in boxes:
        axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                     fc=fc, ec="#7A7672", lw=0.5))
        axa.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                 fontsize=4.0, linespacing=1.2)
    for (x0, y0, x1, y1) in [(3.46, 4.3, 3.72, 4.3), (6.80, 4.9, 7.10, 4.9),
                             (1.6, 4.0, 1.6, 2.55), (3.46, 1.6, 3.72, 1.6)]:
        axa.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                      mutation_scale=7, color="#7A7672",
                                      lw=0.6))
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_a")
