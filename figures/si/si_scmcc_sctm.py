# -*- coding: utf-8 -*-
"""SI Fig. | scMCC vs scTM(两条折叠保真轴解耦)。

(a) 逐链散点:x = scTM(86_sctm_gfix,RhoFold+ 自洽),y = scMCC
    (92_scmcc_gfix,barnaba 标注配对 MCC,annotated 口径,未定义链剔除),
    每方法一色;均值以十字标出。
(b) 逐方法均值双联柱状(scTM / scMCC 并排),含 native 与 rand-match 对照;
    均值断言:native 0.466/0.546、NA-MPNN 0.325/0.463、gRNAde 0.265/0.472、
    PACo 0.263/0.344、RhoDesign 0.240/0.288、randmatch 0.226/0.200。
单独运行:python figures/si/si_scmcc_sctm.py → figures/out/si/si_scmcc_sctm.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label

ORDER = ["native", "NA-MPNN", "gRNAde", "PACo", "RhoDesign", "randmatch"]
EXPECT = {"native": (0.4655, 0.5458), "PACo": (0.2634, 0.3437),
          "RhoDesign": (0.2397, 0.2876), "gRNAde": (0.2651, 0.4724),
          "NA-MPNN": (0.3246, 0.4629), "randmatch": (0.2256, 0.2004)}


def main():
    sctm = C.load("86_sctm_gfix.json")
    scmcc = C.load("92_scmcc_gfix.json")
    for m, (et, em) in EXPECT.items():
        assert abs(sctm["summary"][m]["mean"] - et) < 5e-4, m
        assert abs(scmcc["summary"][m]["scMCC_mean"] - em) < 5e-4, m

    def per_method_vals(m):
        xs, ys = [], []
        tm_key = f"sctm_{m}"
        for stem, row in sctm["per_chain"].items():
            mcc = scmcc["per_chain"].get(stem, {}).get(m, {})
            v = (mcc.get("annotated") or {}).get("MCC")
            if v is None or tm_key not in row:
                continue
            xs.append(row[tm_key]); ys.append(v)
        return np.array(xs), np.array(ys)

    fig, gs = C.make_canvas(66, 1, 2, width_ratios=[1.05, 1.0],
                            wspace=0.30, left=0.065, right=0.99,
                            top=0.88, bottom=0.16)
    # ---- (a) scatter ----
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    from matplotlib.lines import Line2D
    handles = []
    for m in ORDER:
        x, y = per_method_vals(m)
        ax.scatter(x, y, s=2.4, color=C.MCOLOR[m], alpha=0.35, lw=0,
                   zorder=3)
        ax.plot(x.mean(), y.mean(), marker="+", ms=7, mew=1.4,
                color=C.MCOLOR[m], zorder=6)
        handles.append(Line2D([], [], color=C.MCOLOR[m], marker="o", ls="none",
                              ms=3.0, alpha=0.9,
                              label=f"{C.MLAB[m]} (n = {len(x)})"))
    handles.append(Line2D([], [], color="#7A7672", marker="+", ls="none",
                          ms=6, mew=1.4, label="cross = method mean"))
    ax.set_xlabel("scTM (RhoFold+ self-consistency)")
    ax.set_ylabel("scMCC (base-pairing contact fidelity)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.06, 1.06)
    ax.legend(handles=handles, loc="upper left", fontsize=5.0,
              borderaxespad=0.2, labelspacing=0.3, handletextpad=0.3,
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=1.0)
    # ---- (b) mean bars ----
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    xg = np.arange(len(ORDER)); wg = 0.38
    vtm = [sctm["summary"][m]["mean"] for m in ORDER]
    vmc = [scmcc["summary"][m]["scMCC_mean"] for m in ORDER]
    ax.bar(xg - wg / 2, vtm, wg * 0.9, color="#8FA6BB", ec="white", lw=0.3,
           label="scTM", zorder=3)
    ax.bar(xg + wg / 2, vmc, wg * 0.9, color=C.PACO, ec="white", lw=0.3,
           label="scMCC", zorder=3)
    for xi, v in zip(xg - wg / 2, vtm):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=4.6,
                rotation=90, va="bottom")
    for xi, v in zip(xg + wg / 2, vmc):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=4.6,
                rotation=90, va="bottom")
    ax.set_xticks(xg)
    ax.set_xticklabels([C.MLAB[m] for m in ORDER], fontsize=5.0)
    ax.set_ylabel("Mean (193-chain subset)")
    ax.set_ylim(0, 0.68)
    ax.legend(loc="upper right", fontsize=5.2)
    C.save_si(fig, "scmcc_sctm")


if __name__ == "__main__":
    main()
