# -*- coding: utf-8 -*-
"""Fig 1d | 数据漏斗:PDB → RNA-only/复合物两流 → 双重聚类 → 三层测试划分。

PDB 24,252 → 纯 RNA 池 6,988 / 复合物 868;CD-HIT 0.8 + TM ≥ 0.45 传递闭包
→ 300 等价类 → train/validation/test/independent/zero-shot 五路;右侧三层
测试要点框(措辞与正文一致:time-based split、remain unassigned 等)。
单独运行:python fig1_d.py → out/fig1_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import LIGAND, panel_label, save


def draw(fig, gs):
    axd = fig.add_subplot(gs[1, 2:6])
    axd.set_xlim(0, 20); axd.set_ylim(0, 10); axd.axis("off")
    panel_label(axd, "d")
    # 顶层
    C.box(axd, 0.2, 8.35, 5.4, 1.3, C.C_DATA, "PDB RNA entries\n24,252", fs=5.0)
    # 两流
    C.arrow(axd, 2.9, 8.35, 2.9, 7.75, rad=0.0)
    C.box(axd, 0.2, 6.25, 5.4, 1.35, "white",
          "RNA-only pool\n6,988 chains (→ clustering)", fs=4.8, ec="#B7B0A2")
    C.arrow(axd, 5.6, 9.0, 8.2, 9.0)
    C.box(axd, 8.2, 8.35, 5.5, 1.3, C.C_DATA, "RNA–ligand complexes\n868",
          fs=5.0)
    C.arrow(axd, 10.8, 8.35, 10.8, 7.75)
    C.box(axd, 8.2, 6.25, 5.2, 1.35, "white",
          "768 chains enter\nstage-2 finetuning\n(+ stage-1 chains 1:1)",
          fs=4.8, ec="#B7B0A2")
    # 聚类
    C.arrow(axd, 2.9, 6.25, 6.3, 5.55, rad=-0.15)
    C.arrow(axd, 10.8, 6.25, 7.5, 5.55, rad=0.15)
    C.box(axd, 3.6, 3.85, 6.6, 1.6, C.C_SS,
          "CD-HIT 0.8 + TM ≥ 0.45\ntransitive closure →\n"
          "300 equivalence classes", fs=4.8)
    C.arrow(axd, 6.9, 3.85, 6.9, 3.3)
    # 划分行(名称与图注一致,不用缩写)
    splits = [("train", "5,773", C.C_MODEL), ("validation", "96", C.C_MODEL),
              ("test", "477", C.C_MET), ("independent", "235", C.C_MET),
              ("zero-shot", "93", "#D3B5B0")]
    xw = 3.35
    for i, (nm, cnt, fc) in enumerate(splits):
        x0 = 0.55 + i * (xw + 0.28)
        ec = LIGAND if nm == "zero-shot" else C.EC
        C.box(axd, x0, 1.45, xw, 1.5, fc, f"{nm}\n{cnt}", fs=4.9,
              ec=ec, lw=0.9 if nm == "zero-shot" else 0.5,
              bold=(nm == "zero-shot"))
        C.arrow(axd, 6.9, 3.3, x0 + xw / 2, 3.05, rad=0.0 if i == 2 else
                (0.12 if i < 2 else -0.12), lw=0.5)
    axd.text(9.2, 0.55, "class-level 8:1:1 split · 511 chains remain unassigned",
             fontsize=4.7, ha="center", va="center")
    C.box(axd, 14.0, 5.3, 5.8, 4.35, "white",
          "Three test layers\n\n• cluster-based holdout\n   (TM ≥ 0.45)\n"
          "• time-based split (2024+)\n• ligand holdout\n"
          "  (ligands absent\n   from training)", fs=4.7, ec="#B7B0A2")
    return axd


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_d")
