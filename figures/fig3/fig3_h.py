# -*- coding: utf-8 -*-
"""Fig 3h | 骨架质量依赖:x = 骨架 scTM(native 预测 vs 晶体),y = predbb t = 0.1。

AlignIF 与 PACo-Aptamer 双色散点 + OLS 虚线;面板内不写统计文字
(Spearman ρ/P 由图注承载),图例置于右上空区;数据与 ED Fig. 3 同源
(ed_common,86_sctm_gfix)。
单独运行:python fig3_h.py → out/fig3_h.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from scipy import stats

import fig3_common as C
from fig_style import ALIGNIF, PACO, panel_label, save


def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 2:4])
    panel_label(ax, "h")
    sc = C.ED.SCTM["per_chain"]
    x = np.array([sc[s]["sctm_native"] for s in C.STEMS193])
    for m, col in (("AlignIF", ALIGNIF), ("PACo-Aptamer", PACO)):
        pre = C.ED.predbb_recovery(m)
        y = np.array([100 * pre[s][1] for s in C.STEMS193])
        lr = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 30)
        ax.plot(xs, lr.slope * xs + lr.intercept, ls="--", lw=0.9, color=col,
                zorder=2)
        ax.scatter(x, y, s=4.5, color=col, alpha=0.42, lw=0, zorder=3,
                   label=m)
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    ax.set_xlabel("Backbone quality (scTM)")
    ax.set_ylabel("Recovery, t = 0.1 mean (%)")
    ax.set_xlim(0.1, 1.02)
    ax.set_ylim(-4, 105)
    C.tighten_ylabels(fig)
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_h")
