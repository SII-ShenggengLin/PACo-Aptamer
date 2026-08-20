# -*- coding: utf-8 -*-
"""Fig 4c | SS 条件保真 INF:多裁判柱状(native 上界 1.0 / random 下界 0.03)
+ 设计同一性分箱曲线(低同一性下补偿突变保持配对)。

单独运行:python fig4_c.py → out/fig4_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib import gridspec

import fig4_common as C
from fig_style import METHOD_COLOR, NATIVE, NEG, panel_label, save


def draw(fig, gs):
    gsc = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 0:4],
                                           wspace=2.2)
    axc = fig.add_subplot(gsc[0, 0:3])
    panel_label(axc, "c", x=-0.16, y=1.12)
    crows = [("native", "native", NATIVE),
             ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"]),
             ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
             ("AlignIF", "AlignIF", METHOD_COLOR["AlignIF"]),
             ("TACo-Aptamer", "TACo_ens2", METHOD_COLOR["TACo-Aptamer"]),
             ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
             ("random", "random", NEG)]
    names, vals, cis, cols = [], [], [], []
    for lab, key, col in crows:
        d = C.P03[key]["inf"]
        names.append(lab); vals.append(d["mean"])
        cis.append(d["mean"] - d["ci95"][0]); cols.append(col)
    xc = np.arange(len(names))
    axc.bar(xc, vals, 0.62, yerr=cis, color=cols,
            error_kw=dict(lw=0.5, capsize=1.3), ec="white", lw=0.3)
    for x0, v in zip(xc, vals):
        axc.text(x0, v + 0.03, f"{v:.3f}", ha="center", fontsize=4.8,
                 fontweight="bold" if names[int(x0)] in ("native", "random")
                 else "normal")
    axc.set_xticks(xc); axc.set_xticklabels(
        [n.replace("-Aptamer", "") for n in names], rotation=32, ha="right")
    axc.set_ylabel("SS-fidelity INF (input SS vs refolded design)")
    axc.set_ylim(0, 1.14)
    axc.text(0.985, 0.955, "upper bound: native = 1.0\nlower bound: random = 0.03",
             transform=axc.transAxes, ha="right", va="top", fontsize=4.8,
             color="#545450")

    axc2 = fig.add_subplot(gsc[0, 3])
    BIN_KEYS = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"]
    BIN_POS = [0.2, 0.45, 0.55, 0.65, 0.75, 0.9]
    for mlab, mkey in C.MKEY:
        d = C.P03[mkey]["inf_by_identity"]
        ys, xs = [], []
        for bk, bp in zip(BIN_KEYS, BIN_POS):
            if bk in d:
                xs.append(bp); ys.append(d[bk]["inf_mean"])
        axc2.plot(xs, ys, marker="o", ms=2.2, lw=0.8,
                  color=METHOD_COLOR[mlab])
    axc2.set_xlabel("Design identity to native")
    axc2.set_ylabel("INF")
    axc2.set_xlim(0.05, 1.0); axc2.set_ylim(0, 1.02)
    axc2.text(0.08, 0.06, "compensatory mutations:\nhigh INF at low identity",
              fontsize=4.4, color="#545450", transform=axc2.transAxes)
    return axc, axc2


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_c")
