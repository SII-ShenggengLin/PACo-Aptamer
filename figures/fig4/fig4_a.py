# -*- coding: utf-8 -*-
"""Fig 4a | 消融双口径表:全局 best / 界面 best / Δ(界面−整体) / vs full (pp, P)。

300 样本池(3T×100);Full model 行雾蓝底;脚注说明单模型族口径。
单独运行:python fig4_a.py → out/fig4_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig4_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0:3]); axa.axis("off")
    panel_label(axa, "a", x=-0.062, y=1.06)
    rows_abl = [("Full model (ens2)", "baseline"),
                ("– ensemble (single)", "no_ensemble"),
                ("– SS conditioning", "no_ss"),
                ("– data augmentation", "no_aug"),
                ("– ligand context", "no_ligand")]
    hdr = ["Model variant", "Global\nbest (%)", "Interface\nbest (%)",
           "Δ (iface–\noverall, pp)", "Iface Δ vs full\n(pp, P)"]
    cell = []
    for lab, key in rows_abl:
        r = C.P00[key]
        iface_b = 100 * r["iface"]["iface_best"]
        delta_io = 100 * r["iface"]["iface_minus_overall"]
        if key == "baseline":
            pcell = "—"
        else:
            vb = r["vs_baseline_iface"]
            # vb.delta_iface_best = baseline − variant(界面口径)
            dvar = -100 * vb["delta_iface_best"]
            pcell = f"{dvar:+.2f} (P={vb['wilcoxon_p_one_sided']:.3f})" \
                if vb["wilcoxon_p_one_sided"] >= 0.001 else \
                f"{dvar:+.2f} (P={vb['wilcoxon_p_one_sided']:.4f})"
        cell.append([lab, f"{100*r['global_best_477']:.2f}", f"{iface_b:.2f}",
                     f"{delta_io:+.2f}", pcell])
    tb = axa.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(4.9); tb.scale(1, 1.62)
    for (i, j), c in tb.get_celld().items():
        c.set_linewidth(0.4); c.set_edgecolor("#C4C4C0")
        if i == 0:
            c.set_facecolor("#EFEDEA"); c.set_text_props(fontweight="bold")
        elif i == 1:
            c.set_facecolor("#EDF2F7")
    axa.text(0.5, 0.010,
             "300-sample pools (3 T × 100) · final noiface recipe · global = test 477 chains\n"
             "interface = 115 ligand-contact chains (<5 Å; 100 with ≥3 interface residues);\n"
             "rows 2–5 are single models — within that family, component removals cost\n"
             "0.6–0.8 pp globally (interface effects n.s. at this pool size)",
             transform=axa.transAxes, ha="center", fontsize=4.5, color="#545450")
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_a")
