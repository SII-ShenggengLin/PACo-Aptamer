# -*- coding: utf-8 -*-
"""Fig 4d | 每核苷酸恢复率 vs 距配体距离(相对各链自身均值,独立编号面板)。

与 c 同数据同色序的归一化口径(Δ 记号按全图惯例改为文字化轴标题);
零参照虚线入图例("chain mean");不重复方法图例(方法色同 c)。
单独运行:python fig4_d.py → out/fig4_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig4_common as C
from fig_style import METHOD_COLOR, panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "d")
    x = np.arange(5)
    for mlab, mkey in C.MKEY:
        vals = [100 * C.P02[mkey][bn]["norm_mean"] for bn in C.BIN_NAMES]
        lo = [100 * (C.P02[mkey][bn]["norm_mean"] - C.P02[mkey][bn]["norm_ci95"][0])
              for bn in C.BIN_NAMES]
        hi = [100 * (C.P02[mkey][bn]["norm_ci95"][1] - C.P02[mkey][bn]["norm_mean"])
              for bn in C.BIN_NAMES]
        ax.errorbar(x, vals, yerr=[lo, hi], marker="o", ms=2.8, lw=0.9,
                    capsize=1.4, elinewidth=0.5, color=METHOD_COLOR[mlab])
    ax.axhline(0, color="#7A7672", lw=0.6, ls=":")
    ax.legend(handles=[Line2D([], [], color="#7A7672", lw=0.6, ls=":",
                              label="chain mean")],
              loc="upper left", fontsize=6.0, handlelength=1.6,
              handletextpad=0.5, borderaxespad=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(C.BIN_LAB)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-24, 16)
    ax.set_xlabel("Distance to ligand (Å)")
    ax.set_ylabel("Recovery relative to chain mean (%)")
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_d")
