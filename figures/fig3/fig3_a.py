# -*- coding: utf-8 -*-
"""Fig 3a | 三重隔离协议总览:结构簇 8:1:1 → 时间切分(2024+)→ 配体级隔离。

三层嵌套框(米灰/雾蓝/柔橙),层间箭头;底部配体集合与链集合交集为空的声明。
单独运行:python fig3_a.py → out/fig3_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig3_common as C
from fig_style import panel_label, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(fig, gs):
    axa = fig.add_subplot(gs[:, 0:3])
    axa.set_xlim(0, 10); axa.set_ylim(0, 12); axa.axis("off")
    panel_label(axa, "a", x=-0.045, y=1.045)

    # 层1:结构簇
    axa.add_patch(FancyBboxPatch((0.4, 7.6), 9.2, 3.6,
                                 boxstyle="round,pad=0.15", fc="#F0EDE8",
                                 ec="#8C8C88", lw=0.6))
    axa.text(5, 10.75, "1 · Cluster-level holdout", ha="center", fontsize=6.0,
             fontweight="bold", color="#2E5F8A")
    axa.text(5, 9.5, "300 clusters\nCD-HIT 0.8 + TM ≥ 0.45\nclosure → 8 : 1 : 1",
             ha="center", va="center", fontsize=5.2)
    axa.text(0.85, 8.05, "train 5,773 · val 96 · test 477", fontsize=4.9,
             color="#545450")

    # 层2:时间
    axa.add_patch(FancyBboxPatch((0.9, 4.4), 8.2, 2.6,
                                 boxstyle="round,pad=0.15", fc="#DCE6EF",
                                 ec="#8C8C88", lw=0.6))
    axa.text(5, 6.55, "2 · Temporal split", ha="center", fontsize=6.0,
             fontweight="bold", color="#2E5F8A")
    axa.text(5, 5.35, "independent test, n = 235\nchains released 2024+",
             ha="center", va="center", fontsize=5.2)

    # 层3:配体级(高亮橙)
    axa.add_patch(FancyBboxPatch((1.4, 1.2), 7.2, 2.6,
                                 boxstyle="round,pad=0.15", fc="#F6E3DC",
                                 ec="#E8A088", lw=0.9))
    axa.text(5, 3.35, "3 · Ligand-level isolation", ha="center", fontsize=6.0,
             fontweight="bold", color="#B0755F")
    axa.text(5, 2.15, "zero-shot targets, n = 93\nligands unseen during training",
             ha="center", va="center", fontsize=5.2)
    axa.text(5, 0.45,
             "ligand sets: S2 \u2229 zero-shot = empty  ·  chains: S2 \u2229 all test = empty",
             ha="center", fontsize=4.9, color="#B0755F", style="italic")

    for y0, y1 in [(7.6, 7.0), (4.4, 3.8)]:
        axa.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>",
                                      mutation_scale=8, color="#545450",
                                      lw=0.7))
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_a")
