# -*- coding: utf-8 -*-
"""Fig 5b | 新颖性:rank-ordered 最大训练集相似度曲线 + vs AptaDB 中位数刻度。

单独运行:python fig5_b.py → out/fig5_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 2:5])
    panel_label(axb, "b", x=-0.09, y=1.12)
    for m in METHOD_ORDER:
        k = "TACo" if m == "TACo-Aptamer" else m
        rc = np.array(C.NOV["methods"][k]["train"]["rank_curve"])
        xq = np.linspace(0, 1, len(rc))
        axb.plot(xq, rc, lw=0.9, color=METHOD_COLOR[m],
                 label=f"{m} (median {C.NOV['methods'][k]['train']['median']:.2f}, "
                       f"{100*C.NOV['methods'][k]['train']['frac_gt_0.8']:.1f}% > 0.8)")
    axb.axhline(0.8, color="#B07060", lw=0.5, ls=":")
    axb.text(0.99, 0.815, "identity = 0.8", fontsize=4.6, color="#B07060",
             ha="right", va="bottom")
    med_apt = {m: C.NOV["methods"]["TACo" if m == "TACo-Aptamer" else m]["aptadb"]["median"]
               for m in METHOD_ORDER}
    for i, m in enumerate(METHOD_ORDER):
        axb.scatter([1.012], [med_apt[m]], marker="|", s=30, lw=1.0,
                    color=METHOD_COLOR[m], clip_on=False)
    axb.text(1.022, 0.30, "vs\nAptaDB\n(medians)", fontsize=4.4, va="center",
             ha="left", color="#545450", transform=axb.get_yaxis_transform())
    axb.set_xlim(0, 1); axb.set_ylim(0, 1.02)
    axb.set_xlabel("Designs ranked by max identity to training set")
    axb.set_ylabel("Max sequence identity")
    axb.legend(loc="upper left", fontsize=4.9, handlelength=1.4,
               labelspacing=0.25)
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_b")
