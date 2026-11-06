# -*- coding: utf-8 -*-
"""Fig 4c | 每核苷酸恢复率 vs 距配体距离(绝对口径,独立编号面板)。

2026-10-22 起原 c 的双子图拆开各自成面板(c 绝对 / d 归一化),宽度翻倍;
pocket 底纹与图内 "pocket" 文字删除(≤5 分箱刻度自明)。
图例(五方法)只画在 c;d 与 c 相邻、同色同序。
单独运行:python fig4_c.py → out/fig4_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig4_common as C
from fig_style import METHOD_COLOR, panel_label, save


def _series(mkey):
    vals, lo, hi = [], [], []
    for bn in C.BIN_NAMES:
        d = C.P02[mkey][bn]
        vals.append(100 * d["mean"])
        lo.append(100 * (d["mean"] - d["ci95"][0]))
        hi.append(100 * (d["ci95"][1] - d["mean"]))
    return vals, lo, hi


def draw(fig, gs):
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    x = np.arange(5)
    for mlab, mkey in C.MKEY:
        vals, lo, hi = _series(mkey)
        ax.errorbar(x, vals, yerr=[lo, hi], marker="o", ms=2.8, lw=0.9,
                    capsize=1.4, elinewidth=0.5, color=METHOD_COLOR[mlab],
                    label=mlab)
    ax.set_xticks(x)
    ax.set_xticklabels(C.BIN_LAB)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(15, 85)
    ax.set_xlabel("Distance to ligand (Å)")
    ax.set_ylabel("Recovery (%)")
    ax.legend(fontsize=6.0, loc="upper left", ncol=2, handlelength=1.2,
              labelspacing=0.3, columnspacing=0.8, borderaxespad=0.2)
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_c")
