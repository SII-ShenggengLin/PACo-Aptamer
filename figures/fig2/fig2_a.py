# -*- coding: utf-8 -*-
"""Fig 2a | 双口径柱状:best sample(实色)与 t=0.1 mean(半透明)。

柱色=方法(x 轴自明),口径只靠色调区分;图例样块用中性近黑色
(不与任何方法色混淆),标签内注明 solid/translucent;
随机参照虚线一并入图例。统计数字由正文/图注承载。
单独运行:python fig2_a.py → out/fig2_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig2_common as C
from fig_style import (METHOD_COLOR, METHOD_ORDER, NEG, panel_label, save)


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0])
    panel_label(axa, "a")
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    _tk = lambda m: C.main["PACoAptamer" if m == "PACo-Aptamer" else m]
    vb = [100 * _tk(m)["recovery_best_mean"] for m in METHOD_ORDER]
    vt = [100 * _tk(m)["recovery_t01_mean"] for m in METHOD_ORDER]
    axa.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            ec="white", lw=0.3)
    axa.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            alpha=0.45, ec="white", lw=0.3)
    for x, v, vmax in ((xs - w / 2, vb, max(vb)), (xs + w / 2, vt, max(vt))):
        for xi, vi in zip(x, v):
            axa.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", fontsize=4.8,
                     fontweight="bold" if vi == vmax else "normal")
    axa.axhline(25, color=NEG, lw=0.6, ls=":")
    axa.legend(handles=[
        Patch(fc="0.25", ec="none", label="best sample per chain (solid)"),
        Patch(fc="0.25", alpha=0.45, ec="none",
              label="t = 0.1 mean (translucent)"),
        Line2D([], [], color=NEG, lw=0.6, ls=":", label="random")],
        loc="upper right", fontsize=4.8, handlelength=1.4,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    axa.set_xticks(xs); axa.set_xticklabels(METHOD_ORDER, rotation=20,
                                            ha="center", va="top")
    axa.set_xlim(-0.6, 4.6)
    axa.set_ylabel("Sequence recovery (%)"); axa.set_ylim(0, 110)
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_a")
