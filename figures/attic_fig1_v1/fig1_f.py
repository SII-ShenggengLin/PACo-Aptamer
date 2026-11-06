# -*- coding: utf-8 -*-
"""Fig 1f | 训练与评估设置(分组栏样式,参考综述类 evaluation 图)。

四组(浅色标题条 + 虚线分隔):
  1 Two-stage training:S1 预训练 5,773 RNA-only 链(真实渲染 1EHZ)→
    S2 复合物微调 768 链(真实渲染 4YCP)→ 双变体概率集成;
  2 Three test layers:cluster-based 477 / time-based 235 / zero-shot 93;
  3 Compared methods:PACo-Aptamer + 四个同数据重训基线;
  4 Evaluation measures:recovery / interface / foldability / diversity。
坐标系等比。单独运行:python fig1_f.py → out/fig1_f.{pdf,png}
"""
from matplotlib.patches import Circle, Rectangle

import fig1_common as C
from fig_style import ALIGNIF, GRNADE, NAMPNN, RHODESIGN, PACO
from fig_style import panel_label, save

SEP = "#B7B0A2"   # 虚线分隔(暖灰)


def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 0:4])
    ax.set_xlim(0, C.F_LIM[0]); ax.set_ylim(0, C.F_LIM[1])
    ax.set_aspect("equal"); ax.set_axis_off()
    panel_label(ax, "f")

    # ---- 组标题条 + 虚线分隔 ----
    for x, w, t, fc in ((0.5, 13.2, "Two-stage training", "#D9C3A9"),
                        (14.7, 10.0, "Three test layers", "#8FA6BB"),
                        (25.7, 7.8, "Compared methods", "#D3B5B0"),
                        (34.5, 9.0, "Evaluation measures", "#CDBF9E")):
        C.header_bar(ax, x, 9.05, w, 0.8, t, fc)
    for xs in (14.2, 25.2, 34.0):
        ax.plot([xs, xs], [0.8, 8.8], color=SEP, lw=0.8, ls=(0, (1, 2)))

    # ================= 组 1:Two-stage training =================
    C.img_place(ax, "fig1_s1_1.png", 2.3, 6.75, h=2.0)
    C.lab(ax, 4.4, 7.45, "Stage 1 · Pretraining", fs=C.FS_BOX, ha="left")
    C.lab(ax, 4.4, 6.90, "5,773 RNA-only chains", fs=C.FS_TXT, ha="left")
    C.lab(ax, 4.4, 6.40, "Noise + rotations · 60 epochs", fs=C.FS_SM,
          ha="left")
    C.opath(ax, [(2.3, 5.70), (2.3, 5.20)], lw=1.0)
    C.img_place(ax, "fig1_s2.png", 2.3, 4.10, h=2.1)
    C.lab(ax, 4.4, 4.85, "Stage 2 · Finetuning", fs=C.FS_BOX, ha="left")
    C.lab(ax, 4.4, 4.30, "768 RNA–ligand complexes", fs=C.FS_TXT, ha="left")
    C.lab(ax, 4.4, 3.80, "Mixed 1:1 with stage 1 · 8 epochs", fs=C.FS_SM,
          ha="left")
    C.opath(ax, [(2.3, 2.95), (2.3, 2.42)], lw=1.0)
    # 双变体集成
    C.wbox(ax, 0.9, 1.60, 3.6, 0.75, lw=0.8, style="round,pad=0.04")
    C.lab(ax, 2.7, 1.975, "Seed 43", fs=C.FS_SM)
    C.wbox(ax, 0.9, 0.55, 3.6, 0.75, lw=0.8, style="round,pad=0.04")
    C.lab(ax, 2.7, 0.925, "SS dropout 0.15", fs=C.FS_SM)
    C.opath(ax, [(4.5, 1.975), (5.3, 1.975), (5.3, 1.55), (5.6, 1.55)],
            lw=0.8)
    C.opath(ax, [(4.5, 0.925), (5.3, 0.925), (5.3, 1.31), (5.6, 1.31)],
            lw=0.8)
    C.plus(ax, 5.85, 1.43, r=0.24, fs=9)
    C.opath(ax, [(6.09, 1.43), (6.7, 1.43)], lw=0.9)
    C.lab(ax, 6.95, 1.68, "Average stepwise", fs=C.FS_SM, ha="left")
    C.lab(ax, 6.95, 1.18, "probabilities", fs=C.FS_SM, ha="left")
    C.lab(ax, 0.9, 0.22, "Two validation-selected stage-2 variants",
          fs=C.FS_TINY, ha="left")

    # ================= 组 2:Three test layers =================
    for y, name, det in ((7.2, "Cluster-based test set", "477 chains"),
                         (5.0, "Time-based test set",
                          "235 chains · released 2024+"),
                         (2.8, "Zero-shot test set",
                          "93 chains · unseen ligands")):
        C.lab(ax, 16.9, y + 0.30, name, fs=C.FS_BOX, ha="left")
        C.lab(ax, 16.9, y - 0.35, det, fs=C.FS_SM, ha="left")
    # 图标:聚类(三节点互连)/ 时间(时钟)/ 零样本(配体圆点 + 斜杠)
    for dx, dy in ((-0.35, -0.28), (0.35, -0.28), (0.0, 0.32)):
        ax.add_patch(Circle((15.75 + dx, 7.2 + dy), 0.20, fc="white",
                            ec="black", lw=0.7, zorder=3))
    for i, (dx, dy) in enumerate(((-0.35, -0.28), (0.35, -0.28),
                                  (0.0, 0.32))):
        for j, (ex, ey) in enumerate(((-0.35, -0.28), (0.35, -0.28),
                                      (0.0, 0.32))):
            if j > i:
                ax.plot([15.75 + dx, 15.75 + ex], [7.2 + dy, 7.2 + ey],
                        color="black", lw=0.5, zorder=2)
    ax.add_patch(Circle((15.75, 5.0), 0.42, fc="white", ec="black", lw=0.8,
                        zorder=3))
    ax.plot([15.75, 15.75], [5.0, 5.30], color="black", lw=0.8, zorder=4)
    ax.plot([15.75, 15.95], [5.0, 5.14], color="black", lw=0.8, zorder=4)
    ax.add_patch(Circle((15.75, 2.8), 0.38, fc=C.C_LIG, ec="black", lw=0.8,
                        zorder=3))
    ax.plot([15.30, 16.20], [2.35, 3.25], color="black", lw=1.0, zorder=4)

    # ================= 组 3:Compared methods =================
    methods = [("PACo-Aptamer", PACO), ("RhoDesign", RHODESIGN),
               ("AlignIF", ALIGNIF), ("NA-MPNN", NAMPNN), ("gRNAde", GRNADE)]
    for k, (name, fc) in enumerate(methods):
        y = 7.5 - k * 1.2
        ax.add_patch(Rectangle((26.15, y - 0.22), 0.5, 0.5, fc=fc,
                               ec="black", lw=0.7, zorder=3))
        C.lab(ax, 26.95, y + 0.03, name, fs=C.FS_TXT, ha="left")
    C.lab(ax, 25.95, 1.35, "All retrained on identical data", fs=C.FS_TINY,
          ha="left", style="italic")

    # ================= 组 4:Evaluation measures =================
    rows = [(7.4, "Sequence recovery", "Best sample · t = 0.1 mean"),
            (5.6, "Interface recovery", "Positions < 5 Å from ligand"),
            (3.8, "Foldability", "scTM (RhoFold+)"),
            (2.0, "Diversity & novelty", "Against native sequences")]
    for y, name, det in rows:
        C.lab(ax, 36.4, y + 0.30, name, fs=C.FS_BOX, ha="left")
        C.lab(ax, 36.4, y - 0.35, det, fs=C.FS_SM, ha="left")
    # 图标:recovery(三根柱)/ interface(口袋圆点)/ foldability(同心圆)/
    # diversity(散布点)
    for k, h in enumerate((0.30, 0.55, 0.80)):
        ax.add_patch(Rectangle((35.05 + k * 0.32, 7.4 - 0.30), 0.26, h,
                               fc=C.C_RNA_L, ec="black", lw=0.6, zorder=3))
    ax.add_patch(Circle((35.5, 5.6), 0.38, fc="white", ec="black", lw=0.8,
                        zorder=3))
    ax.add_patch(Circle((35.5, 5.6), 0.12, fc=C.C_LIG, ec="black", lw=0.5,
                        zorder=4))
    ax.add_patch(Circle((35.5, 3.8), 0.42, fc="white", ec="black", lw=0.8,
                        zorder=3))
    ax.add_patch(Circle((35.5, 3.8), 0.20, fc=C.C_RNA_L, ec="black", lw=0.6,
                        zorder=4))
    for dx, dy in ((-0.30, 0.22), (0.18, 0.34), (0.36, -0.10),
                   (-0.12, -0.30), (0.28, -0.38), (-0.38, -0.05)):
        ax.add_patch(Circle((35.5 + dx, 2.0 + dy), 0.075, fc="black",
                            ec="none", zorder=3))
    return ax


if __name__ == "__main__":
    import fig1_common as C
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_f")
