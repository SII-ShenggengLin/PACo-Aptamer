# -*- coding: utf-8 -*-
"""Fig 2f | 链内 Spearman ρ 分布直方图:正值石板蓝 / 负值灰玫红。

取代旧版单一大提琴+抖动(作者反馈太丑):0 为分箱边界,正负分色,
零线虚线、黑色均值线;84% 为正由色块面积直接呈现,数字在图注。
单独运行:python fig2_f.py → out/fig2_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.patches import Patch

import fig2_common as C
from fig_style import NEG, PACO, panel_label, save

BINS = np.arange(-0.85, 0.951, 0.05)


def draw(fig, gs):
    axf = fig.add_subplot(gs[1, 2])
    panel_label(axf, "f")
    counts, edges = np.histogram(C.within, bins=BINS)
    share = 100 * counts / len(C.within)
    centers = (edges[:-1] + edges[1:]) / 2
    pos = centers > 0
    axf.bar(centers[pos], share[pos], width=0.045, color=PACO,
            ec="white", lw=0.3)
    axf.bar(centers[~pos], share[~pos], width=0.045, color=NEG,
            ec="white", lw=0.3)
    axf.axvline(0, color="#7A7672", lw=0.6, ls=":")
    axf.axvline(float(np.mean(C.within)), color="black", lw=0.9)
    axf.set_xlim(-0.85, 0.9)
    axf.set_ylim(0, 13.5)
    axf.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    axf.legend(handles=[Patch(fc=PACO, ec="none", label="ρ > 0"),
                        Patch(fc=NEG, ec="none", label="ρ < 0")],
               loc="upper left", fontsize=4.8, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    axf.set_xlabel("Within-chain Spearman ρ")
    axf.set_ylabel("Chains (%)")
    return axf


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_f")
