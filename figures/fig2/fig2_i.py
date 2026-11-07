"""Fig 2i | 按配体分层的恢复率优势:PACo-Aptamer − RhoDesign(t = 0.1 口径)。"""
import numpy as np

import fig2_common as C
from fig_style import NEG, PACO, load, panel_label, save

SHORT = {
    "purine_riboswitch": "Purine", "theophylline_aptamer": "Theophylline",
    "xanthine_riboswitch": "Xanthine", "ztp_riboswitch": "ZTP",
    "dG_riboswitch": "dG", "mango3_aptamer": "Mango-III",
    "fam_1MMS": "1MMS", "fam_3GCA": "3GCA", "fam_8HB1": "8HB1",
}

def _rows():
    d = load("family_stratified_recovery.json")
    rng = np.random.RandomState(0)
    rows = []
    for k, g in d["groups"].items():
        if k == "other":
            continue
        tv = g["PACo_vs_RhoDesign"]
        deltas = np.array([v["delta_t01"]
                           for v in tv["per_chain_delta"].values()])
        bs = [rng.choice(deltas, len(deltas), replace=True).mean()
              for _ in range(2000)]
        rows.append((SHORT[k], 100 * tv["delta_t01"]["mean"],
                     100 * np.percentile(bs, 2.5),
                     100 * np.percentile(bs, 97.5)))
    rows.sort(key=lambda r: -r[1])
    return rows

def draw(fig, gs):
    axi = fig.add_subplot(gs[2, 2])
    panel_label(axi, "i")
    rows = _rows()
    n = len(rows)
    axi.axvline(0, color="#7A7672", lw=0.6, ls=":")
    for rank, (nm, mean, lo, hi) in enumerate(rows):
        y = n - 1 - rank
        c = PACO if mean > 0 else NEG
        axi.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o",
                     ms=3.0, color=c, ecolor=c, elinewidth=0.7,
                     capsize=1.4, capthick=0.7, zorder=3)
    axi.set_yticks(range(n))
    axi.set_yticklabels([r[0] for r in rows], fontsize=4.2)
    axi.tick_params(axis="y", pad=0.6, length=1.2, width=0.4)
    axi.set_ylim(-0.6, n - 0.4)
    axi.set_xlim(-6, 16)
    axi.set_xticks([-5, 0, 5, 10, 15])
    axi.set_xlabel("Recovery advantage over RhoDesign,\nt = 0.1 mean (%)")
    return axi

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_i")
