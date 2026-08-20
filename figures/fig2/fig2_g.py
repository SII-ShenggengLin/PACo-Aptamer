# -*- coding: utf-8 -*-
"""Fig 2g | 界面恢复率:绝对(左轴,实色)+ 归一化 Δ(右轴,半透明)双口径。

≤5 Å 界面残基;95% CI 误差棒;TACo vs RhoDesign 界面 Wilcoxon(110 共享链)。
单独运行:python fig2_g.py → out/fig2_g.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig2_common as C
from fig_style import METHOD_COLOR, load, panel_label, save

IFACE_KEY = "≤5Å(界面)"


def draw(fig, gs):
    axg1 = fig.add_subplot(gs[2, 5:8])
    panel_label(axg1, "g", x=-0.135, y=1.12)
    p02 = load("77_p02_distance_gradient.json")["methods"]
    mrows = ["TACo_ens2", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]
    mlab = ["TACo", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]
    iv = [100 * p02[m][IFACE_KEY]["mean"] for m in mrows]
    err = [100 * (p02[m][IFACE_KEY]["mean"] - p02[m][IFACE_KEY]["ci95"][0])
           for m in mrows]
    nm = [100 * p02[m][IFACE_KEY]["norm_mean"] for m in mrows]
    nme = [100 * (p02[m][IFACE_KEY]["norm_mean"] - p02[m][IFACE_KEY]["norm_ci95"][0])
           for m in mrows]
    xg = np.arange(5); wg = 0.38
    cols = [METHOD_COLOR[m] for m in mlab]
    axg1.bar(xg - wg / 2, iv, wg, yerr=err, error_kw=dict(lw=0.5, capsize=1.2),
             color=cols, ec="white", lw=0.3)
    for x, v in zip(xg - wg / 2, iv):
        axg1.text(x, v + 4.5, f"{v:.1f}", ha="center", fontsize=4.7,
                  fontweight="bold" if v == max(iv) else "normal")
    axg1.set_xticks(xg); axg1.set_xticklabels(mlab, rotation=30, ha="right")
    axg1.set_ylabel("Interface recovery (%)"); axg1.set_ylim(0, 95)
    axg2 = axg1.twinx()
    axg2.bar(xg + wg / 2, nm, wg, yerr=nme, error_kw=dict(lw=0.5, capsize=1.2),
             color=cols, alpha=0.45, ec="white", lw=0.3)
    axg2.axhline(0, color="#545450", lw=0.5)
    axg2.set_ylabel("Δ (interface − chain mean, pp)", fontsize=5.6)
    axg2.set_ylim(-26, 13)
    axg2.spines["right"].set_visible(True); axg2.spines["right"].set_linewidth(0.5)
    axg2.spines["top"].set_visible(False)
    axg1.text(0.985, 0.97, "solid: absolute (left axis)\nfaded: normalized (right axis)",
              transform=axg1.transAxes, fontsize=4.8, ha="right", va="top",
              color="#545450",
              bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.5))
    _ifaceW = load("19_iface_wilcoxon.json")
    axg1.annotate((f"TACo vs RhoDesign: Δ = +{100*(_ifaceW['taco_mean']-_ifaceW['rho_mean']):.1f} pp, "
                   f"P = {_ifaceW['p']:.2f} (n.s.)\n"
                   "(paired Wilcoxon, 110 shared interface chains)"),
                  xy=(0.5, 0.10), xycoords="axes fraction", fontsize=4.8,
                  ha="center", color="#2E5F8A")
    return axg1, axg2


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_g")
