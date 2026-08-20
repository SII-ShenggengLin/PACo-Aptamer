# -*- coding: utf-8 -*-
"""Fig 1e | 主结果预览 mini-panel:splits_v2 test 477 链五方法 pool-in-best 柱状。

数据:btest_v2_main_table.json(与 Fig 2c 同源,此处仅作预览)。
单独运行:python fig1_e.py → out/fig1_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig1_common as C
from fig_style import (METHOD_COLOR, METHOD_ORDER, panel_label, save)


def draw(fig, gs):
    axe = fig.add_subplot(gs[1, 6:8])
    panel_label(axe, "e", x=-0.24, y=1.12)
    tk = lambda m: C.MAIN["TACoAptamer" if m == "TACo-Aptamer" else m]
    vb = [100 * tk(m)["recovery_best_mean"] for m in METHOD_ORDER]
    xe = np.arange(len(METHOD_ORDER))
    axe.bar(xe, vb, 0.62, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            ec="white", lw=0.3)
    for x0, v in zip(xe, vb):
        axe.text(x0, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8,
                 fontweight="bold" if v == max(vb) else "normal")
    axe.set_xticks(xe)
    axe.set_xticklabels([m.replace("-Aptamer", "") for m in METHOD_ORDER],
                        rotation=32, ha="right")
    axe.set_ylabel("Recovery, pool-in-best (%)")
    axe.set_title("splits_v2 test (n = 477)", fontsize=5.2, pad=3)
    axe.set_ylim(0, 92)
    axe.text(0.97, 0.03, "preview of Fig. 2", transform=axe.transAxes,
             fontsize=4.4, ha="right", color="#8C8C88", style="italic")
    return axe


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_e")
