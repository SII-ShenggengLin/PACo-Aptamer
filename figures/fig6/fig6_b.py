# -*- coding: utf-8 -*-
"""Fig 6b | 逐位恢复序列条(8HZD_A:native + 5 方法最优设计)。

行 = native + 各方法 best sample;列 = 序列位置;填色 = 与 native 一致;
顶部橙色带 = 界面位置(任一原子 <5 Å of N618/O2I,链 A 残基 7/8/14/15/28/29/30);
右侧两列 = 整体恢复率 % 与界面位恢复率 %。
单独运行:python fig6_b.py → out/fig6_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save

import matplotlib.pyplot as plt


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 7:16])
    panel_label(axb, "c", x=-0.075, y=1.10)
    rows = [("native", C.natA, None, None)] + [
        (m, C.SEQS[m][0], C.SEQS[m][1], None) for m in METHOD_ORDER]
    L = len(C.natA)
    iface_pos = [C.RESIDS.index(rn) for rn in C.IFACE]   # 0 基序列位置

    def iface_id(seq):
        return np.mean([seq[p] == C.natA[p] for p in iface_pos])

    for ri, (name, sq, bv, _) in enumerate(rows):
        y0 = len(rows) - 1 - ri
        if name == "native":
            for pi, ch_ in enumerate(sq):
                axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                         fontsize=4.9, fontweight="bold", color="#3D5A45")
            axb.text(-1.2, y0 + 0.5, "native", ha="right", va="center",
                     fontsize=5.2, fontweight="bold", color="#3D5A45")
            continue
        col = METHOD_COLOR[name]
        for pi, ch_ in enumerate(sq):
            match = ch_ == C.natA[pi]
            axb.add_patch(plt.Rectangle((pi, y0), 1, 1,
                                        fc=col if match else "white",
                                        ec="#E4E2DE", lw=0.3))
            axb.text(pi + 0.5, y0 + 0.5, ch_, ha="center", va="center",
                     fontsize=4.6, color="white" if match else "#8C8C88")
        axb.text(-1.2, y0 + 0.5, name.replace("-Aptamer", ""), ha="right",
                 va="center", fontsize=5.2, color=col)
        fi = iface_id(sq)
        axb.text(L + 1.0, y0 + 0.5, f"{100*bv:.0f}%", ha="left",
                 va="center", fontsize=5.0,
                 fontweight="bold" if name == "TACo-Aptamer" else "normal",
                 color=col)
        axb.text(L + 3.6, y0 + 0.5, f"{100*fi:.0f}%", ha="left",
                 va="center", fontsize=5.0,
                 fontweight="bold" if name == "TACo-Aptamer" else "normal",
                 color=col)
    for pi in iface_pos:
        axb.add_patch(plt.Rectangle((pi, len(rows) - 0.1), 1, 0.42, fc="#E8A088",
                                    ec="none"))
    axb.text(np.mean(iface_pos) + 0.5, len(rows) + 0.55,
             f"interface positions (< {C.CONTACT:.0f} Å of N618)",
             ha="center", fontsize=4.6, color="#B0755F")
    axb.text(L + 1.0, len(rows) - 0.55, "identity", ha="left", fontsize=4.4,
             color="#545450")
    axb.text(L + 3.6, len(rows) - 0.55, "iface", ha="left", fontsize=4.4,
             color="#545450")
    axb.set_xlim(-6.5, L + 6.5)
    axb.set_ylim(-0.3, len(rows) + 1.15)
    axb.axis("off")
    axb.set_title("Best sample designs, 8HZD_A (filled = match to native)",
                  fontsize=5.4, pad=8)
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig6_c")
    for m in METHOD_ORDER:
        s, v = C.SEQS[m]
        ii = np.mean([s[p] == C.natA[p] for p in
                      [C.RESIDS.index(rn) for rn in C.IFACE]])
        print(f"{m:14s} overall {100*v:5.1f}%  iface {100*ii:5.1f}%")
