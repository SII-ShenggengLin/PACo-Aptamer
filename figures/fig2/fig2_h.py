"""Fig 2h | 长度分层箱线:≤40 / 40–80 / >80 nt 三档 × 五方法恢复率。"""
import numpy as np
from matplotlib.lines import Line2D

import fig2_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save

def draw(fig, gs):
    axh = fig.add_subplot(gs[2, 1])
    panel_label(axh, "h")
    lbins = [(0, 40, "≤40 nt"), (40, 80, "40–80 nt"), (80, 10000, ">80 nt")]
    pos, data, colors = [], [], []
    for mi, m in enumerate(METHOD_ORDER):
        for bi, (lo, hi, _) in enumerate(lbins):
            ss = [C.pc[m][s] for s in C.STEMS if lo <= C.chain_len[s] < hi]
            data.append([100 * v for v in ss]); pos.append(bi + (mi - 2) * 0.16)
            colors.append(METHOD_COLOR[m])
    bp = axh.boxplot(data, positions=pos, widths=0.13, whis=1.5,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=0.7),
                     boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                     capprops=dict(lw=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.9)
    for mi in range(len(METHOD_ORDER)):
        idx = [3 * mi + bi for bi in range(3)]
        means = [np.mean(data[i]) for i in idx]
        axh.scatter([pos[i] for i in idx], means, s=4, color="black", zorder=5,
                    lw=0)
    axh.set_xticks(range(3)); axh.set_xticklabels([b[2] for b in lbins])
    axh.set_ylabel("Recovery, best sample (%)")
    axh.set_xlabel("Native chain length")
    axh.set_ylim(0, 122); axh.set_yticks([0, 25, 50, 75, 100])
    handles = [Line2D([], [], marker="s", ls="", color=METHOD_COLOR[m],
                      markersize=3.5, label=m) for m in METHOD_ORDER]
    handles.append(Line2D([], [], marker="o", ls="", color="black",
                          markersize=3, label="method mean"))
    axh.legend(handles=handles, ncol=3, loc="upper center", fontsize=4.8,
               handletextpad=0.25, columnspacing=0.7, labelspacing=0.25)
    return axh

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_h")
