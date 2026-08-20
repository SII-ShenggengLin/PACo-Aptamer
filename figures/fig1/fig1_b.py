# -*- coding: utf-8 -*-
"""Fig 1b | 数据漏斗:PDB → RNA-only/复合物两流 → 双重聚类 → 三重隔离划分。

PDB 24,252 → S1 预训练 6,988 / S2 微调 868;CD-HIT 0.8 + TM ≥ 0.45 传递闭包
→ 300 簇 → train/val/test/indep/zs 五路;右侧 Quasi-blind 协议要点框。
单独运行:python fig1_b.py → out/fig1_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 3:8])
    axb.set_xlim(0, 20); axb.set_ylim(0, 10); axb.axis("off")
    panel_label(axb, "b", x=-0.028, y=1.10)
    # 顶层
    C.box(axb, 0.2, 8.35, 5.4, 1.3, C.C_DATA, "PDB RNA entries\n24,252", fs=5.0)
    # 两流
    C.arrow(axb, 2.9, 8.35, 2.9, 7.75, rad=0.0)
    C.box(axb, 0.2, 6.25, 5.4, 1.35, "white",
          "RNA-only chains 6,988\n→ S1 pretraining", fs=4.8, ec="#C4C4C0")
    C.arrow(axb, 5.6, 9.0, 8.2, 9.0)
    C.box(axb, 8.2, 8.35, 5.2, 1.3, C.C_DATA, "RNA–ligand complexes\n868",
          fs=5.0)
    C.arrow(axb, 10.8, 8.35, 10.8, 7.75)
    C.box(axb, 8.2, 6.25, 5.2, 1.35, "white",
          "→ S2 finetuning\n(canonical 6,346 merged set)", fs=4.8, ec="#C4C4C0")
    # 聚类
    C.arrow(axb, 2.9, 6.25, 6.3, 5.35, rad=-0.15)
    C.arrow(axb, 10.8, 6.25, 7.5, 5.35, rad=0.15)
    C.box(axb, 3.6, 3.95, 6.6, 1.35, C.C_SS,
          "CD-HIT 0.8 + TM ≥ 0.45\ntransitive closure → 300 clusters", fs=4.8)
    C.arrow(axb, 6.9, 3.95, 6.9, 3.3)
    # 划分行
    splits = [("train", "5,773", C.C_MODEL), ("val", "96", C.C_MODEL),
              ("test", "477", C.C_MET), ("indep", "235", C.C_MET),
              ("zs", "93", "#F6E3DC")]
    xw = 3.35
    for i, (nm, cnt, fc) in enumerate(splits):
        x0 = 0.55 + i * (xw + 0.28)
        ec = "#E8A088" if nm == "zs" else C.EC
        C.box(axb, x0, 1.45, xw, 1.5, fc, f"{nm}\n{cnt}", fs=4.9,
              ec=ec, lw=0.9 if nm == "zs" else 0.5,
              bold=(nm == "zs"))
        C.arrow(axb, 6.9, 3.3, x0 + xw / 2, 3.05, rad=0.0 if i == 2 else
                (0.12 if i < 2 else -0.12), lw=0.5)
    axb.text(9.2, 0.55, "cluster-wise 8 : 1 : 1 · 511 chains discarded",
             fontsize=4.7, color="#545450", ha="center", va="center")
    axb.annotate("ligand-level isolation", xy=(16.7, 1.45), xytext=(16.7, 0.55),
                 fontsize=4.7, color="#B0755F", ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#B0755F"))
    C.box(axb, 14.0, 5.3, 5.8, 4.35, "white",
          "Quasi-blind protocol\n\n• cluster holdout (TM ≥ 0.45)\n"
          "• temporal split (2024+)\n• ligand-level isolation\n"
          "  (test ligands unseen)", fs=4.7, ec="#C4C4C0")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_b")
