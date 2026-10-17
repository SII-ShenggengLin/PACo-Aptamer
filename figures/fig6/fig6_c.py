# -*- coding: utf-8 -*-
"""Fig 6c | 三条案例链的 best sample 恢复率(8HZD_A/B、8HZE_A ∈ indep ∩ zs)。

五方法并排柱状,TACo 柱顶标数值。数据:69_alayer_gfix.json。
单独运行:python fig6_c.py → out/fig6_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, TACO_D, load, panel_label, save


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:3])
    panel_label(axc, "c", x=-0.20, y=1.12)
    al = load("69_alayer_gfix.json")["per_chain"]
    xc = np.arange(len(C.CASE))
    wc = 0.15
    for mi, m in enumerate(METHOD_ORDER):
        key = "TACoAptamer" if m == "TACo-Aptamer" else m
        vb = [100 * al[s][key]["recovery_best"] for s in C.CASE]
        axc.bar(xc + (mi - 2) * wc, vb, wc * 0.92, color=METHOD_COLOR[m],
                ec="white", lw=0.2, label=m.replace("-Aptamer", ""))
    for mi, m in enumerate(METHOD_ORDER):
        key = "TACoAptamer" if m == "TACo-Aptamer" else m
        for ci, s in enumerate(C.CASE):
            v = 100 * al[s][key]["recovery_best"]
            if m == "TACo-Aptamer":
                axc.text(ci + (mi - 2) * wc, v + 1.0, f"{v:.0f}", ha="center",
                         fontsize=4.2, color=TACO_D, fontweight="bold")
    axc.set_xticks(xc)
    axc.set_xticklabels(C.CASE, fontsize=5.2)
    axc.set_ylabel("Recovery, best sample (%)")
    axc.set_ylim(0, 100)
    axc.legend(fontsize=4.6, ncol=2, loc="lower left", handletextpad=0.3,
               columnspacing=0.7)
    axc.set_title("case chains: independent test + zero-shot",
                  fontsize=5.4, pad=3)
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig6_c")
