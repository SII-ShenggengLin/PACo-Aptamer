# -*- coding: utf-8 -*-
"""Fig 3a | 三重隔离协议总览:结构簇 8:1:1 → 时间切分(2024+)→ 配体级隔离。

三层嵌套框(沙/浅石板蓝/灰玫红),层间箭头;底部无泄漏声明(措辞与正文
time-based split / never occur in any training complex 一致)。
单独运行:python fig3_a.py → out/fig3_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import LIGAND, panel_label, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(fig, gs):
    axa = fig.add_subplot(gs[0:2, 0:3])
    axa.set_xlim(0, 10); axa.set_ylim(0, 12); axa.axis("off")
    panel_label(axa, "a")

    # 层1:结构簇
    axa.add_patch(FancyBboxPatch((0.4, 7.6), 9.2, 3.6,
                                 boxstyle="round,pad=0.15", fc="#D9C3A9",
                                 ec="#7A7672", lw=0.6))
    axa.text(5, 10.75, "1 · Cluster-based holdout", ha="center", fontsize=6.0,
             fontweight="bold")
    axa.text(5, 9.5, "300 equivalence classes\nCD-HIT 0.8 + TM ≥ 0.45\nclosure → 8:1:1 split",
             ha="center", va="center", fontsize=5.2)
    axa.text(0.85, 8.05, "train 5,773 · validation 96 · test 477",
             fontsize=4.9)

    # 层2:时间
    axa.add_patch(FancyBboxPatch((0.9, 4.4), 8.2, 2.6,
                                 boxstyle="round,pad=0.15", fc="#8FA6BB",
                                 ec="#7A7672", lw=0.6))
    axa.text(5, 6.55, "2 · Time-based split", ha="center", fontsize=6.0,
             fontweight="bold")
    axa.text(5, 5.35, "independent test set, n = 235\nchains released 2024 or later",
             ha="center", va="center", fontsize=5.2)

    # 层3:配体级(高亮陶土橙)
    axa.add_patch(FancyBboxPatch((1.4, 1.2), 7.2, 2.6,
                                 boxstyle="round,pad=0.15", fc="#D3B5B0",
                                 ec=LIGAND, lw=0.9))
    axa.text(5, 3.35, "3 · Ligand holdout", ha="center", fontsize=6.0,
             fontweight="bold")
    axa.text(5, 2.15, "zero-shot test set, n = 93\nligands absent from training",
             ha="center", va="center", fontsize=5.2)
    axa.text(5, 0.45,
             "no test chain occurs in training or finetuning;\n"
             "no zero-shot ligand occurs in any training complex",
             ha="center", fontsize=4.9, style="italic")

    for y0, y1 in [(7.6, 7.0), (4.4, 3.8)]:
        axa.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>",
                                      mutation_scale=8, color="#7A7672",
                                      lw=0.7))
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_a")
