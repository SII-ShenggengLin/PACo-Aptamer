# -*- coding: utf-8 -*-
"""Fig 4g | recovery-matched 负对照逐链 Δ 分布:scTM(TACo) − scTM(randmatch)。

均值 +3.8 pp(95% CI [2.4, 5.1],单侧 Wilcoxon P = 4×10⁻⁶)。
单独运行:python fig4_g.py → out/fig4_g.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig4_common as C
from fig_style import TACO, panel_label, save


def draw(fig, gs):
    axg = fig.add_subplot(gs[1, 6:8])
    panel_label(axg, "g", x=-0.19, y=1.12)
    diff = np.array([v["sctm_TACo"] - v["sctm_randmatch"]
                     for v in C.SCTM["per_chain"].values()
                     if "sctm_TACo" in v and "sctm_randmatch" in v]) * 100
    axg.hist(diff, bins=36, color=TACO, alpha=0.85, ec="white", lw=0.2)
    axg.axvline(0, color="#545450", lw=0.6)
    axg.axvline(float(np.mean(diff)), color="#2E5F8A", lw=0.9, ls="--")
    axg.text(0.955, 0.92,
             f"mean Δ = {np.mean(diff):+.1f} pp\n95% CI [2.4, 5.1]\n"
             "P = 4 × 10$^{{-6}}$ (one-sided Wilcoxon)\n\n"
             "recovery-matched random collapses to\ngRNAde level (0.226 vs 0.219)",
             transform=axg.transAxes, ha="right", va="top", fontsize=5.0,
             color="#2E5F8A")
    axg.set_xlabel("scTM(TACo) − scTM(recovery-matched random), pp")
    axg.set_ylabel("Chains")
    return axg


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_g")
    print("randmatch diff mean:", np.mean(
        np.array([v["sctm_TACo"] - v["sctm_randmatch"]
                  for v in C.SCTM["per_chain"].values()
                  if "sctm_TACo" in v and "sctm_randmatch" in v]) * 100).round(2))
