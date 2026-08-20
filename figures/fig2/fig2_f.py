# -*- coding: utf-8 -*-
"""Fig 2f | 逐靶 Δ 双向条形:TACo − RhoDesign,477 链按 Δ 排序。

正向雾蓝/负向灰;中位数虚线;橙色方块(底部)标配体界面链。
单独运行:python fig2_f.py → out/fig2_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig2_common as C
from fig_style import TACO, load, panel_label, save


def draw(fig, gs):
    axf = fig.add_subplot(gs[2, 0:5])
    panel_label(axf, "f", x=-0.075, y=1.12)
    delta = np.array([C.pc["TACo-Aptamer"][s] - C.pc["RhoDesign"][s]
                      for s in C.STEMS]) * 100
    order = np.argsort(delta)
    cc = load("74_chain_ccd.json")
    iface_flag = np.array([s in cc for s in C.STEMS])[order]
    dv = delta[order]
    axf.bar(np.arange(len(dv))[dv >= 0], dv[dv >= 0], width=1.0, color=TACO,
            alpha=0.85, lw=0)
    axf.bar(np.arange(len(dv))[dv < 0], dv[dv < 0], width=1.0, color="#A8A8A4",
            alpha=0.9, lw=0)
    axf.scatter(np.flatnonzero(iface_flag), np.full(int(iface_flag.sum()), -37.5),
                marker="s", s=1.6, c="#E8A088", lw=0, clip_on=False)
    axf.axhline(0, color="#545450", lw=0.6)
    med = float(np.median(delta))
    axf.axhline(med, color=TACO, lw=0.7, ls=":")
    axf.text(len(dv) * 0.998, med + 1.8, f"median {med:+.1f} pp", ha="right",
             fontsize=5.0, color=TACO)
    axf.set_xlim(-5, len(dv) + 5); axf.set_ylim(-45, 64)
    axf.set_xlabel(f"Chains ranked by per-chain Δ recovery (n = {len(dv)})")
    axf.set_ylabel("Δ recovery vs\nRhoDesign (pp)")
    axf.text(0.015, 0.035,
             f"{100*np.mean(delta > 0):.0f}% of chains positive · orange squares = ligand-interface chains",
             transform=axf.transAxes, fontsize=5.0, color="#545450",
             bbox=dict(fc="white", ec="none", pad=1.0))
    return axf


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_f")
