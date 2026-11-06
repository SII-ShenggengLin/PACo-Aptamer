# -*- coding: utf-8 -*-
"""ED Fig. 2 | Ligand-family leave-one-out and coverage(LOFO)。

(a) 五个配体家族剔除重训的 zs best Δ(vs 全量基线 73.35%),按 |Δ| 降序;
    full_ctl 同分布重训作噪声参照(暖灰),灰带 = ±0.04% 重训噪声;
    柱下标 Δ 与 P,x 轴标家族剔除链数(面板内不再放角标说明,噪声带含义进图注)。
(b) 家族覆盖曲线:仅保留前 k 个家族的训练链数 vs zs best;
    参考线 = 全量训练 73.35%(虚线)+ 同宽噪声带;参考值与链数进图注。
轴标签精简为单词短语(Δ 记号不进图),与主图 Fig. 2/3 轴标签风格一致。
单独运行:python figures/ed/ed_fig2.py → figures/out/ed_fig2.{pdf,png}
"""
import numpy as np

import ed_common as C
from fig_style import PACO, panel_label, save

VAR_ORDER = ["aminoglycoside", "sam_sah", "2ba", "ntp", "cla"]  # |Δ| 降序
VAR_LAB = {"aminoglycoside": "aminoglycoside", "sam_sah": "SAM/SAH",
           "2ba": "2BA", "ntp": "NTP", "cla": "chlorophyll a"}


def draw_a(ax):
    panel_label(ax, "a")
    zt = C.LOFO["zs_table"]
    ds = C.LOFO["datasets"]
    keys = [f"s2_v2_lofo_{f}" for f in VAR_ORDER]
    dv = [100 * zt[k]["vs_baseline"]["delta_best_mean"] for k in keys]
    pv = [zt[k]["vs_baseline"]["wilcoxon_p_best"] for k in keys]
    nr = [ds[f"lofo_{f}"]["removed"] for f in VAR_ORDER]
    ctl_d = 100 * zt["s2_v2_full_ctl"]["vs_baseline"]["delta_best_mean"]
    ctl_p = zt["s2_v2_full_ctl"]["vs_baseline"]["wilcoxon_p_best"]

    xg = np.arange(6)
    vals = dv + [ctl_d]
    cols = [PACO] * 5 + ["#B7B0A2"]
    ax.axhspan(-0.04, 0.04, color="#7A7672", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color="#7A7672", lw=0.6)
    ax.bar(xg, vals, 0.62, color=cols, ec="white", lw=0.3, zorder=3)
    for x, v, p in zip(xg, vals, pv + [ctl_p]):
        if v <= 0:
            ax.text(x, v - 0.10, f"{v:+.1f}%", ha="center", va="top",
                    fontsize=5.0)
            ax.text(x, v - 0.48, C.pfmt_sci(p), ha="center",
                    va="top", fontsize=5.0)
        else:
            ax.text(x, v + 0.10, f"{v:+.2f}%", ha="center", va="bottom",
                    fontsize=5.0)
            ax.text(x, v + 0.44, C.pfmt_sci(p), ha="center",
                    va="bottom", fontsize=5.0)
    labs = [f"{VAR_LAB[f]}\n(−{n} chains)" for f, n in zip(VAR_ORDER, nr)]
    labs.append("control\n(all complexes)")
    ax.set_xticks(xg)
    ax.set_xticklabels(labs, fontsize=5.2)
    ax.set_ylabel("Change in zero-shot recovery (%)")
    ax.set_ylim(-2.9, 1.15)


def draw_b(ax):
    panel_label(ax, "b")
    cc = C.LOFO["coverage_curve"]
    ks = [c["k"] for c in cc]
    ys = [100 * c["zs_best"] for c in cc]
    full = 100 * C.LOFO["full_reference"]["zs_best"]
    xg = np.arange(len(ks))
    ax.axhspan(full - 0.04, full + 0.04, color="#7A7672", alpha=0.10, lw=0,
               zorder=0)
    ax.axhline(full, color="#7A7672", lw=0.8, ls="--", zorder=1)
    ax.plot(xg, ys, "-o", color=PACO, ms=3.2, lw=1.0, mew=0.4, mec="white",
            zorder=3)
    for x, y in zip(xg, ys):
        dy = 0.13 if x != 2 else -0.30
        ax.text(x, y + dy, f"{y:.1f}", ha="center", fontsize=5.0)
    ax.text(-0.32, full + 0.10, "full training", fontsize=5.0, ha="left",
            va="bottom")
    ax.set_xticks(xg)
    ax.set_xticklabels([str(k) for k in ks], fontsize=5.2)
    ax.set_xlabel("Ligand families kept in training")
    ax.set_ylabel("Zero-shot recovery (%)")
    ax.set_ylim(70.4, 74.6)
    ax.set_xlim(-0.4, 3.4)


def main():
    fig, gs = C.make_canvas(66, 1, 2, width_ratios=[1.32, 1.0],
                            wspace=0.34, left=0.062, right=0.995,
                            top=0.86, bottom=0.235)
    draw_a(fig.add_subplot(gs[0, 0]))
    draw_b(fig.add_subplot(gs[0, 1]))
    save(fig, "ed_fig2")


if __name__ == "__main__":
    main()
