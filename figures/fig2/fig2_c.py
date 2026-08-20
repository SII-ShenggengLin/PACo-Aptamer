# -*- coding: utf-8 -*-
"""Fig 2c | 双口径柱状:pool-in-best(实色)与 t=0.1 mean(半透明)+ Wilcoxon P。

vs RhoDesign 配对 Wilcoxon 结果白底框标注于面板右上。
单独运行:python fig2_c.py → out/fig2_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig2_common as C
from fig_style import (METHOD_COLOR, METHOD_ORDER, panel_label, save)


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:2])
    panel_label(axc, "c", x=-0.24, y=1.1)
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    _tk = lambda m: C.main["TACoAptamer" if m == "TACo-Aptamer" else m]
    vb = [100 * _tk(m)["recovery_best_mean"] for m in METHOD_ORDER]
    vt = [100 * _tk(m)["recovery_t01_mean"] for m in METHOD_ORDER]
    axc.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            ec="white", lw=0.3)
    axc.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            alpha=0.45, ec="white", lw=0.3)
    for x, v in zip(xs - w / 2, vb):
        axc.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=4.8,
                 fontweight="bold" if v == max(vb) else "normal")
    for x, v in zip(xs + w / 2, vt):
        axc.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=4.8, alpha=0.75)
    axc.axhline(25, color="#B07060", lw=0.6, ls=":")
    axc.text(4.4, 26.5, "random", fontsize=4.6, color="#B07060", ha="right")
    pR = C.wil["B_test_477"]["RhoDesign"]["recovery_best"]
    pRt = C.wil["B_test_477"]["RhoDesign"]["recovery_t01_mean"]
    axc.text(0.98, 0.985,
             (f"vs RhoDesign (paired Wilcoxon, n = 477)\n"
              f"best: Δ = +{100*pR['delta']:.1f} pp, {C.pfmt_p(pR['p_two_sided'])}\n"
              f"t = 0.1: Δ = +{100*pRt['delta']:.1f} pp, {C.pfmt_p(pRt['p_two_sided'])}"),
             transform=axc.transAxes, fontsize=4.9, ha="right", va="top",
             bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.6))
    axc.set_xticks(xs); axc.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    axc.set_ylabel("Sequence recovery (%)"); axc.set_ylim(0, 122)
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_c")
