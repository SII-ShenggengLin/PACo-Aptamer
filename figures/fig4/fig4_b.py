"""Fig 4b | 界面恢复率柱状:五方法 ≤5 Å 界面位置,柱顶 95% CI。"""
import numpy as np

import fig4_common as C
from fig_style import METHOD_COLOR, panel_label, save

IFACE_KEY = "≤5Å(界面)"

def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 1])
    panel_label(axb, "b")
    mlab = [m for m, _ in C.MKEY]
    mrow = [k for _, k in C.MKEY]
    iv = [100 * C.P02[k][IFACE_KEY]["mean"] for k in mrow]
    err_lo = [100 * (C.P02[k][IFACE_KEY]["mean"] - C.P02[k][IFACE_KEY]["ci95"][0])
              for k in mrow]
    err_hi = [100 * (C.P02[k][IFACE_KEY]["ci95"][1] - C.P02[k][IFACE_KEY]["mean"])
              for k in mrow]
    xg = np.arange(len(mlab))
    axb.bar(xg, iv, 0.62, yerr=[err_lo, err_hi],
            color=[METHOD_COLOR[m] for m in mlab],
            error_kw=dict(lw=0.5, capsize=1.4), ec="white", lw=0.3)
    for x, v, e in zip(xg, iv, err_hi):
        axb.text(x, v + e + 5.5, f"{v:.1f}", ha="center", fontsize=5.2)
    axb.set_xticks(xg)
    axb.set_xticklabels(mlab, rotation=20, ha="center", va="top")
    axb.set_xlim(-0.6, len(mlab) - 0.4)
    axb.set_ylabel("Interface recovery (%)")
    axb.set_ylim(0, 82)
    return axb

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_b")
