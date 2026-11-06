# -*- coding: utf-8 -*-
"""SI Fig. | Predicted-backbone per-chain transfer(预测骨架逐链分析)。

(a) 每方法 crystal vs predicted backbone 逐链 best sample 散点
    (5 个 mini-panel,n = 193,y = x 参考线);标注 Δbest 与 Δt01(pp)。
    crystal = 63_main_table_gfix per_chain;predicted = 94/101/103/100/102
    号 predbb per_chain。Δ 断言:PACo −5.3 / RhoDesign −6.9 / gRNAde −1.6 /
    AlignIF −15.7 / NA-MPNN −19.1 pp。
(b) 32 条 zero-shot ∩ predbb 重叠链专题(105_predbb_zs.json):
    PACo-Aptamer−RhoDesign 优势 Δbest +6.7pp(P = 4.8×10⁻⁶)保持 vs
    Δt01 −0.07pp(P = 0.42)归零。
(c) 骨架切换效应(predbb − crystal):t01 口径 PACo-Aptamer −11.0pp vs
    RhoDesign −6.3pp(best 口径 −5.7 / −6.4pp)。
b/c 各占底行一半,间距由底层 wspace 控制;RhoFold+ 来源写在图注。
单独运行:python figures/si/si_predbb.py → figures/out/si/si_predbb.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label

PRED_FILE = {"PACoAptamer": "94_predbb_paco.json",
             "RhoDesign": "101_rhodesign_predbb.json",
             "gRNAde": "103_grnade_predbb.json",
             "AlignIF": "100_alignif_predbb.json",
             "NA-MPNN": "102_nampnn_predbb.json"}
EXPECT_DBEST = {"PACoAptamer": -5.31, "RhoDesign": -6.92, "gRNAde": -1.57,
                "AlignIF": -15.74, "NA-MPNN": -19.08}


def draw_scatter(ax, m, cry, pred, letter=None):
    if letter:
        panel_label(ax, letter)
    common = sorted(set(cry) & set(pred))
    x = np.array([100 * cry[k]["recovery_best"] for k in common])
    y = np.array([100 * pred[k]["recovery_best"] for k in common])
    dt = np.mean([100 * (pred[k]["recovery_t01_mean"]
                         - cry[k]["recovery_t01_mean"]) for k in common])
    db = (y - x).mean()
    assert abs(db - EXPECT_DBEST[m]) < 0.02, (m, db)
    lim = (0, 102)
    ax.plot(lim, lim, ls="--", lw=0.6, color="#7A7672", zorder=1)
    ax.scatter(x, y, s=3.0, color=C.MCOLOR[m], alpha=0.5, lw=0, zorder=3)
    ax.text(0.04, 0.96, f"best sample: {db:+.1f} pp\nt = 0.1: {dt:+.1f} pp",
            transform=ax.transAxes, fontsize=5.0, va="top",
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    ax.set_title(C.MLAB[m], fontsize=6.0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Crystal backbone, best sample (%)", fontsize=5.2)
    ax.set_ylabel("Predicted backbone, best sample (%)" if m == "PACoAptamer"
                  else "", fontsize=5.2)
    ax.tick_params(labelsize=4.8)
    ax.set_aspect("equal")


def draw_zs(ax1, ax2):
    d = C.load("105_predbb_zs.json")
    zs = d["zs_subset"]["delta_PACo_minus_RhoDesign"]
    assert zs["best"]["n"] == 32
    assert abs(zs["best"]["mean"] - 0.06706) < 5e-5
    assert abs(zs["t01"]["mean"] - (-0.00068)) < 5e-5
    panel_label(ax1, "b")
    xg = np.arange(2)
    vals = [100 * zs["best"]["mean"], 100 * zs["t01"]["mean"]]
    ps = [zs["best"]["wilcoxon_p_two_sided"], zs["t01"]["wilcoxon_p_two_sided"]]
    los = [100 * (zs[k]["mean"] - zs[k]["ci95"][0]) for k in ("best", "t01")]
    his = [100 * (zs[k]["ci95"][1] - zs[k]["mean"]) for k in ("best", "t01")]
    ax1.bar(xg, vals, 0.5, color=[C.PACO, C.PACO_D], ec="white",
            lw=0.3, yerr=[los, his],
            error_kw=dict(lw=0.6, capsize=1.5), zorder=3)
    for xi, v, h, p in zip(xg, vals, his, ps):
        ax1.text(xi, v + h + 0.35, f"{v:+.2f} pp\n{C.pfmt_sci(p)}",
                 ha="center", fontsize=5.0, va="bottom")
    ax1.axhline(0, color="#7A7672", lw=0.6)
    ax1.set_xticks(xg)
    ax1.set_xticklabels(["best sample", "t = 0.1 mean"], fontsize=5.6)
    ax1.set_ylabel("PACo-Aptamer − RhoDesign (pp)", fontsize=6.0)
    ax1.set_ylim(-4.5, 12.5)
    ax1.tick_params(labelsize=5.6)
    # 右:骨架切换效应分解
    panel_label(ax2, "c")
    dec = d["zs_subset_crystal_decomp"]["predbb_minus_crystal"]
    xg = np.arange(2)
    for j, (key, off) in enumerate([("best", -0.19), ("t01", 0.19)]):
        vals = [100 * dec["PACoAptamer"][key]["mean"],
                100 * dec["RhoDesign"][key]["mean"]]
        cols = [C.PACO if j == 0 else C.PACO_D, C.RHODESIGN]
        ax2.bar(xg + off, vals, 0.34, color=cols, ec="white", lw=0.3,
                zorder=3, label="best sample" if j == 0 else "t = 0.1 mean")
        for xi, v in zip(xg + off, vals):
            ax2.text(xi, v - 0.45, f"{v:+.1f} pp",
                     ha="center", va="top", fontsize=5.0)
    ax2.axhline(0, color="#7A7672", lw=0.6)
    ax2.set_xticks(xg)
    ax2.set_xticklabels(["PACo-Aptamer", "RhoDesign"], fontsize=5.6)
    ax2.set_ylabel("Predicted − crystal (pp)", fontsize=6.0)
    ax2.set_ylim(-16.5, 2.5)
    ax2.tick_params(labelsize=5.6)
    ax2.legend(loc="lower left", fontsize=5.0)


def main():
    main63 = C.load("63_main_table_gfix.json")["per_chain"]
    pred = {m: C.load(f)["per_chain"] for m, f in PRED_FILE.items()}
    fig, gs = C.make_canvas(112, 2, 1, height_ratios=[1.0, 0.86],
                            hspace=0.55, left=0.060, right=0.995,
                            top=0.90, bottom=0.11)
    from matplotlib import gridspec
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0],
                                              wspace=0.30)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                              wspace=0.42)
    for i, m in enumerate(C.METHODS):
        draw_scatter(fig.add_subplot(gs_top[0, i]), m, main63[m], pred[m],
                     letter="a" if i == 0 else None)
    draw_zs(fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1]))
    C.save_si(fig, "predbb")


if __name__ == "__main__":
    main()
