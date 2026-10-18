# -*- coding: utf-8 -*-
"""Fig 1d | 数据漏斗:PDB → RNA-only/复合物两流 → 双重聚类 → 三重隔离划分。

PDB 24,252 → 纯 RNA 池 6,988 / 复合物 868;CD-HIT 0.8 + TM ≥ 0.45 传递闭包
→ 300 簇 → train/val/test/indep/zs 五路;右侧三层隔离协议要点框。
单独运行:python fig1_d.py → out/fig1_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    axd = fig.add_subplot(gs[1, 2:6])
    axd.set_xlim(0, 20); axd.set_ylim(0, 10); axd.axis("off")
    panel_label(axd, "d", x=-0.035, y=1.10)
    # 顶层
    C.box(axd, 0.2, 8.35, 5.4, 1.3, C.C_DATA, "PDB RNA entries\n24,252", fs=5.0)
    # 两流
    C.arrow(axd, 2.9, 8.35, 2.9, 7.75, rad=0.0)
    C.box(axd, 0.2, 6.25, 5.4, 1.35, "white",
          "RNA-only pool\n6,988 chains (→ clustering)", fs=4.8, ec="#C4C4C0")
    C.arrow(axd, 5.6, 9.0, 8.2, 9.0)
    C.box(axd, 8.2, 8.35, 5.2, 1.3, C.C_DATA, "RNA–ligand complexes\n868",
          fs=5.0)
    C.arrow(axd, 10.8, 8.35, 10.8, 7.75)
    C.box(axd, 8.2, 6.25, 5.2, 1.35, "white",
          "768 chains → S2 finetuning\n(+ S1 train chains 1:1)", fs=4.8, ec="#C4C4C0")
    # 聚类
    C.arrow(axd, 2.9, 6.25, 6.3, 5.35, rad=-0.15)
    C.arrow(axd, 10.8, 6.25, 7.5, 5.35, rad=0.15)
    C.box(axd, 3.6, 3.95, 6.6, 1.35, C.C_SS,
          "CD-HIT 0.8 + TM ≥ 0.45\ntransitive closure → 300 equivalence classes", fs=4.8)
    C.arrow(axd, 6.9, 3.95, 6.9, 3.3)
    # 划分行
    splits = [("train", "5,773", C.C_MODEL), ("val", "96", C.C_MODEL),
              ("test", "477", C.C_MET), ("indep", "235", C.C_MET),
              ("zs", "93", "#F6E3DC")]
    xw = 3.35
    for i, (nm, cnt, fc) in enumerate(splits):
        x0 = 0.55 + i * (xw + 0.28)
        ec = "#E8A088" if nm == "zs" else C.EC
        C.box(axd, x0, 1.45, xw, 1.5, fc, f"{nm}\n{cnt}", fs=4.9,
              ec=ec, lw=0.9 if nm == "zs" else 0.5,
              bold=(nm == "zs"))
        C.arrow(axd, 6.9, 3.3, x0 + xw / 2, 3.05, rad=0.0 if i == 2 else
                (0.12 if i < 2 else -0.12), lw=0.5)
    axd.text(9.2, 0.55, "cluster-wise 8 : 1 : 1 · 511 chains discarded",
             fontsize=4.7, color="#545450", ha="center", va="center")
    axd.annotate("ligand holdout", xy=(16.7, 1.45), xytext=(16.7, 0.55),
                 fontsize=4.7, color="#B0755F", ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#B0755F"))
    C.box(axd, 14.0, 5.3, 5.8, 4.35, "white",
          "Three-layer isolation\n\n• cluster-based holdout (TM ≥ 0.45)\n"
          "• temporal split (2024+)\n• ligand holdout\n"
          "  (test ligands unseen)", fs=4.7, ec="#C4C4C0")
    return axd


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_d")
