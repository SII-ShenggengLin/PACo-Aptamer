# -*- coding: utf-8 -*-
"""Fig 2a | 评估协议流程:条件输入 → 池采样 → 双口径恢复 → 统计检验。

单独运行:python fig2_a.py → out/fig2_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_DATA, C_MODEL, C_MET = "#F0EDE8", "#DCE6EF", "#E9E2D8"


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0:3]); axa.set_xlim(0, 10); axa.set_ylim(0, 6.4)
    axa.axis("off")
    panel_label(axa, "a", x=-0.045, y=1.14)
    boxes = [
        (0.1, 4.0, 3.0, 1.85, C_DATA, "Conditioning input:\nRNA backbone + SS\n(backbone-only parity)"),
        (3.55, 4.0, 3.0, 1.85, C_MODEL, "Sequence pool\n3 T × 100\nper method"),
        (7.0, 4.0, 2.9, 1.85, C_MET, "Recovery:\nbest / t = 0.1\n+ interface (<5 Å)"),
        (0.1, 0.7, 3.0, 1.85, C_MODEL, "Same-data\nretraining\n(5 methods)"),
        (3.55, 0.7, 6.35, 1.85, C_MET, "Per-chain paired Wilcoxon + 95% CI bootstrap\nvs strongest baseline"),
    ]
    for x, y, w, h, fc, txt in boxes:
        axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                     fc=fc, ec="#8C8C88", lw=0.5))
        axa.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                 fontsize=5.2)
    for (x0, y0, x1, y1) in [(3.1, 4.9, 3.55, 4.9), (6.55, 4.9, 7.0, 4.9),
                             (1.6, 4.0, 1.6, 2.55), (3.1, 1.6, 3.55, 4.0)]:
        axa.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                      mutation_scale=7, color="#545450",
                                      lw=0.6))
    axa.text(5, 6.15, "Quasi-blind evaluation protocol", ha="center",
             fontsize=6.4, style="italic", color="#2E5F8A")
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_a")
