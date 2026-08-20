# -*- coding: utf-8 -*-
"""Fig 2i | 长度分层箱线:≤40 / 40–80 / >80 nt 三档 × 五方法恢复率。

箱体方法色;黑点+连线 = 方法均值。单独运行:python fig2_i.py →
out/fig2_i.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig2_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save


def draw(fig, gs):
    axi = fig.add_subplot(gs[3, 5:8])
    panel_label(axi, "i", x=-0.135, y=1.1)
    lbins = [(0, 40, "≤40 nt"), (40, 80, "40–80 nt"), (80, 10000, ">80 nt")]
    pos, data, colors = [], [], []
    for mi, m in enumerate(METHOD_ORDER):
        for bi, (lo, hi, _) in enumerate(lbins):
            ss = [C.pc[m][s] for s in C.STEMS if lo <= C.chain_len[s] < hi]
            data.append([100 * v for v in ss]); pos.append(bi + (mi - 2) * 0.16)
            colors.append(METHOD_COLOR[m])
    bp = axi.boxplot(data, positions=pos, widths=0.13, whis=1.5,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=0.7),
                     boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                     capprops=dict(lw=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.9)
    for mi in range(len(METHOD_ORDER)):
        idx = [3 * mi + bi for bi in range(3)]
        means = [np.mean(data[i]) for i in idx]
        axi.scatter([pos[i] for i in idx], means, s=4, color="black", zorder=5,
                    lw=0)
        axi.plot([pos[i] for i in idx], means, color="black", lw=0.5, alpha=0.5)
    axi.set_xticks(range(3)); axi.set_xticklabels([b[2] for b in lbins])
    axi.set_ylabel("Recovery, pool-in-best (%)")
    axi.set_xlabel("Native chain length")
    handles = [Line2D([], [], marker="s", ls="", color=METHOD_COLOR[m],
                      markersize=3.5, label=m) for m in METHOD_ORDER]
    axi.legend(handles=handles, ncol=2, loc="lower left", fontsize=4.6,
               handletextpad=0.3, columnspacing=0.8)
    axi.text(0.985, 0.97, "dots = method means", transform=axi.transAxes,
             ha="right", va="top", fontsize=4.6, color="#545450")
    return axi


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_i")
