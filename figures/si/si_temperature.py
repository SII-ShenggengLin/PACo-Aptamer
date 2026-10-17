# -*- coding: utf-8 -*-
"""SI Fig. | Temperature–diversity scan(温度扫描)。

(a) 分温度池 mean recovery(t 口径均值)随温度 0.1/0.5/1.0 变化;
(b) 分温度池 diversity(逐链池内平均两两差异)与 unique fraction。
TACo(n = 477)/ RhoDesign / gRNAde(n = 437,均为簇级测试集可评估链)。
数据源 data/survey/82_temp_diversity_gfix.json(gRNAde 为解码修正后重算)。
注:源文件只有 per-temperature 池均值;best sample 为跨温度 300 池指标
(63_main_table_gfix),不在本图分温度口径内,故未画。
单独运行:python figures/si/si_temperature.py → figures/out/si/si_temperature.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label

TEMPS = ["t=0.1", "t=0.5", "t=1.0"]
XT = [0.1, 0.5, 1.0]
EXPECT = {("TACo", "t=0.1", "rec"): 0.5972, ("RhoDesign", "t=0.1", "rec"): 0.5463,
          ("gRNAde", "t=0.1", "rec"): 0.3596,
          ("TACo", "t=1.0", "div"): 0.3141, ("gRNAde", "t=1.0", "uniq"): 1.0}


def main():
    d = C.load("82_temp_diversity_gfix.json")
    for (m, t, k), ref in EXPECT.items():
        assert abs(d[m][t][k]["mean"] - ref) < 5e-4, (m, t, k)
    fig, gs = C.make_canvas(58, 1, 2, wspace=0.30, left=0.070, right=0.99,
                            top=0.88, bottom=0.17)
    # (a) recovery
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", x=-0.16, y=1.08)
    for m in ["TACo", "RhoDesign", "gRNAde"]:
        y = [100 * d[m][t]["rec"]["mean"] for t in TEMPS]
        ax.plot(XT, y, "-o", ms=3.2, lw=1.1, color=C.MCOLOR[m],
                label=f"{m} (n = {d[m]['t=0.1']['rec']['n']})", zorder=3)
        for xi, yi in zip(XT, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=4.4,
                        color=C.MCOLOR[m])
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Pool mean recovery (%)")
    ax.set_xticks(XT)
    ax.set_ylim(25, 66)
    ax.legend(loc="lower left", fontsize=4.8, borderaxespad=0.1)
    # (b) diversity + uniq
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.16, y=1.08)
    for m in ["TACo", "RhoDesign", "gRNAde"]:
        yd = [d[m][t]["div"]["mean"] for t in TEMPS]
        yu = [d[m][t]["uniq"]["mean"] for t in TEMPS]
        ax.plot(XT, yd, "-o", ms=3.2, lw=1.1, color=C.MCOLOR[m], zorder=3,
                label=f"{m} diversity")
        ax.plot(XT, yu, ":s", ms=2.6, lw=0.8, color=C.MCOLOR[m], alpha=0.65,
                zorder=2, label=f"{m} unique fraction")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Pool diversity / unique fraction")
    ax.set_xticks(XT)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=4.4,
              borderaxespad=0.0, labelspacing=0.3)
    C.save_si(fig, "temperature")


if __name__ == "__main__":
    main()
