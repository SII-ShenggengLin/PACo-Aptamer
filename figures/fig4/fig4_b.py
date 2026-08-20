# -*- coding: utf-8 -*-
"""Fig 4b | 每核苷酸恢复率 vs 距配体距离(绝对 + 归一化双子图)。

≤5 Å pocket 区柔橙底纹;五方法莫兰迪色;误差棒 95% CI。
单独运行:python fig4_b.py → out/fig4_b.{pdf,png}(与拼合版物理尺寸一致)
"""
from matplotlib import gridspec

import fig4_common as C
from fig_style import METHOD_COLOR, panel_label, save


def draw(fig, gs):
    gsb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 3:8],
                                           wspace=1.0)
    for sub, (mode, ylab, letter_off) in enumerate([
            ("abs", "Recovery (%)", ""), ("norm", "Δ recovery vs chain mean (pp)", "")]):
        axb = fig.add_subplot(gsb[0, sub])
        if sub == 0:
            panel_label(axb, "b", x=-0.115, y=1.12)
        for mlab, mkey in C.MKEY:
            vals, lo, hi = [], [], []
            for bn in C.BIN_NAMES:
                d = C.P02[mkey][bn]
                vals.append(100 * d["mean"])
                lo.append(100 * (d["mean"] - d["ci95"][0]))
                hi.append(100 * (d["ci95"][1] - d["mean"]))
            if mode == "norm":
                vals = [100 * C.P02[mkey][bn]["norm_mean"] for bn in C.BIN_NAMES]
                lo = [100 * (C.P02[mkey][bn]["norm_mean"] - C.P02[mkey][bn]["norm_ci95"][0])
                      for bn in C.BIN_NAMES]
                hi = [100 * (C.P02[mkey][bn]["norm_ci95"][1] - C.P02[mkey][bn]["norm_mean"])
                      for bn in C.BIN_NAMES]
            axb.errorbar(C.BIN_X, vals, yerr=[lo, hi], marker="o", ms=2.4,
                         lw=0.9, capsize=1.2, elinewidth=0.5,
                         color=METHOD_COLOR[mlab],
                         label=mlab if sub == 0 else None)
        axb.axvspan(0, 5, color="#E8A088", alpha=0.14, lw=0)
        if sub == 0:
            axb.text(2.5, 88, "pocket", fontsize=4.8, ha="center",
                     color="#B0755F")
            axb.set_ylim(0, 92)
        else:
            axb.axhline(0, color="#545450", lw=0.5)
            axb.set_ylim(-20, 14)
            axb.text(2.5, 11.5, "pocket", fontsize=4.8, ha="center",
                     color="#B0755F")
        axb.set_xticks(C.BIN_X); axb.set_xticklabels(C.BIN_LAB)
        axb.set_xlabel("Distance to nearest ligand atom (Å)")
        axb.set_ylabel(ylab)
        if sub == 0:
            axb.legend(fontsize=4.9, loc="lower left", handlelength=1.2,
                       labelspacing=0.25, ncol=2, columnspacing=0.8,
                       frameon=True, facecolor="white", framealpha=0.85,
                       edgecolor="none")   # gRNAde 低曲线穿图例区 → 白底衬
        if sub == 1:
            axb.text(0.97, 0.05,
                     "absolute (left) · normalized (right)\nmethod level removed",
                     transform=axb.transAxes, fontsize=4.4, color="#545450",
                     ha="right", va="bottom")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_b")
