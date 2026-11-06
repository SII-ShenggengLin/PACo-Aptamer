# -*- coding: utf-8 -*-
"""ED Fig. 1 | Chemical novelty stratification(化学新颖性分层)。

(a) 87 条 zero-shot 链:配体对 247 个训练配体的 max Tanimoto vs 逐链优势(%),
    按相似度分箱着色(四箱取 12 色板浅→深);OLS 回归线 + Spearman ρ/P。
(b) 分箱优势均值柱状:y 轴从 0 起,柱上标均值,n 并入横轴刻度第二行
    (双侧 Wilcoxon P 移入图注,[0,0.2) n=4 不检验)。
(c) 指纹稳健性:Morgan/MACCS/Topological 三指纹的 ρ(优势)分组柱状,
    best 实色、t = 0.1 半透明(主图两口径约定),无星号、无 n.s.。
轴标签与 SI 图 6("Maximum Tanimoto similarity")和主图 Fig. 3i
("Advantage over RhoDesign")同款,Δ 记号不进图。
单独运行:python figures/ed/ed_fig1.py → figures/out/ed_fig1.{pdf,png}
"""
import numpy as np
from scipy import stats

import ed_common as C
from fig_style import PACO, panel_label, save

BIN_EDGES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0001)]
BIN_LAB = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–1.0"]
# 12 色板浅→深四档(沙浅→暖灰→浅蓝→石板蓝),相似度越高越深
BIN_COLS = ["#D9C3A9", "#B7B0A2", "#8FA6BB", "#6B7B8D"]


def _bin_idx(x):
    for i, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= x < hi:
            return i
    return len(BIN_EDGES) - 1


def draw_a(ax):
    panel_label(ax, "a")
    sc = C.T93["scatter"]
    x = np.array([s["max_tanimoto"] for s in sc])
    y = np.array([100 * s["delta_t01"] for s in sc])
    for i in range(4):
        m = np.array([_bin_idx(v) == i for v in x])
        ax.scatter(x[m], y[m], s=5.5, color=BIN_COLS[i], lw=0, alpha=0.85,
                   label=f"{BIN_LAB[i]} (n = {m.sum()})", zorder=3)
    lr = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, lr.slope * xs + lr.intercept, ls="--", lw=0.9, color="#7A7672", zorder=2)
    ax.axhline(0, color="#7A7672", lw=0.5, zorder=1)
    sp = C.T93["spearman"]["sim_vs_delta_t01"]
    ax.text(0.03, 0.04,
            f"Spearman ρ = {sp['rho']:.2f}\n{C.pfmt_sci(sp['p'])}, n = {len(x)}",
            transform=ax.transAxes, fontsize=5.0, va="bottom",
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))
    ax.set_xlabel("Maximum Tanimoto similarity")
    ax.set_ylabel("Advantage over RhoDesign\nt = 0.1 mean (%)")
    ax.set_xlim(0.10, 1.06)
    ax.set_ylim(-14, 24)
    ax.legend(loc="upper right", fontsize=5.0, handletextpad=0.15,
              borderaxespad=0.2, labelspacing=0.25)


def draw_b(ax):
    panel_label(ax, "b")
    bt = C.T93["bins_table"]
    means = [100 * b["delta_best_PACo_minus_RhoDesign"]["mean"] for b in bt]
    ns = [b["n"] for b in bt]
    xg = np.arange(4)
    ax.bar(xg, means, 0.62, color=BIN_COLS, ec="white", lw=0.3)
    for x, v in zip(xg, means):
        ax.text(x, v + 0.25, f"+{v:.1f}%", ha="center", fontsize=5.0)
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{lab}\nn = {n}" for lab, n in zip(BIN_LAB, ns)],
                       fontsize=5.2)
    ax.set_ylabel("Advantage over RhoDesign\nbest sample (%)")
    ax.set_ylim(0, 7.8)


def draw_c(ax):
    panel_label(ax, "c")
    sp = C.R106["spearman"]
    fps = ["morgan", "maccs", "rdkit_topo"]
    flab = ["Morgan", "MACCS", "Topological"]
    r_best = [sp[f]["sim_vs_delta_best"]["rho"] for f in fps]
    r_t01 = [sp[f]["sim_vs_delta_t01"]["rho"] for f in fps]
    xg = np.arange(3)
    wg = 0.36
    ax.bar(xg - wg / 2, r_best, wg, color=PACO, ec="white", lw=0.3,
           label="best sample")
    ax.bar(xg + wg / 2, r_t01, wg, color=PACO, alpha=0.42, ec="white", lw=0.3,
           label="t = 0.1 mean")
    ax.axhline(0, color="#7A7672", lw=0.6)
    for x, v in zip(xg - wg / 2, r_best):
        ax.text(x, v - 0.024, f"{v:.2f}", ha="center", va="top", fontsize=5.0)
    for x, v in zip(xg + wg / 2, r_t01):
        ax.text(x, v - 0.024, f"{v:.2f}", ha="center", va="top", fontsize=5.0)
    ax.set_xticks(xg)
    ax.set_xticklabels(flab, fontsize=5.2)
    ax.set_ylabel("Spearman ρ\n(similarity vs advantage)")
    ax.set_ylim(-0.55, 0.14)
    ax.legend(loc="upper left", fontsize=5.0, borderaxespad=0.1,
              handlelength=1.2, labelspacing=0.3)


def main():
    fig, gs = C.make_canvas(66, 1, 3, width_ratios=[1.10, 0.98, 1.0],
                            wspace=0.46, left=0.065, right=0.982,
                            top=0.88, bottom=0.20)
    draw_a(fig.add_subplot(gs[0, 0]))
    axb = fig.add_subplot(gs[0, 1])
    draw_b(axb)
    draw_c(fig.add_subplot(gs[0, 2]))
    save(fig, "ed_fig1")


if __name__ == "__main__":
    main()
