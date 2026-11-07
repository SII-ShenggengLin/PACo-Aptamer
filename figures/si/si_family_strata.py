"""SI Fig. | Ligand-family stratification(配体家族分层,t=0.1 口径)。"""
import numpy as np

import si_common as C
from fig_style import panel_label

PRETTY = {"fam_1MMS": "1MMS", "theophylline_aptamer": "Theophylline",
          "purine_riboswitch": "Purine", "fam_3GCA": "3GCA",
          "dG_riboswitch": "dG", "ztp_riboswitch": "ZTP",
          "mango3_aptamer": "Mango-III", "xanthine_riboswitch": "Xanthine",
          "fam_8HB1": "8HB1", "other": "Other"}
ORDER = ["purine_riboswitch", "theophylline_aptamer", "mango3_aptamer",
         "ztp_riboswitch", "xanthine_riboswitch", "fam_1MMS", "fam_3GCA",
         "dG_riboswitch", "fam_8HB1", "other"]

LAB = dict(C.MLAB, PACoAptamer="PACo-Aptamer")

def main():
    rows = {r["family"]: r for r in
            C.load_csv("tables/family_strata.csv")}

    t = rows["theophylline_aptamer"]
    assert abs(float(t["PACoAptamer_t01"]) - 0.772161) < 1e-9
    assert abs(float(t["RhoDesign_t01"]) - 0.695495) < 1e-9
    assert abs(float(t["gRNAde_t01"]) - 0.501117) < 1e-9
    assert abs(float(t["NA-MPNN_t01"]) - 0.734659) < 1e-9
    assert abs(float(t["AlignIF_t01"]) - 0.656895) < 1e-9
    j = C.load("family_stratified_recovery.json")
    for fam in ORDER:
        assert j["groups"][fam]["n_chains"] == int(rows[fam]["n"]), fam

    fig, gs = C.make_canvas(66, 1, 1, left=0.055, right=0.995,
                            top=0.86, bottom=0.20)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    xg = np.arange(len(ORDER))
    wg = 0.17
    for i, m in enumerate(C.METHODS):
        vals = [100 * float(rows[f][f"{m}_t01"]) for f in ORDER]
        ax.bar(xg + (i - 2) * wg, vals, wg * 0.92, color=C.MCOLOR[m],
               ec="white", lw=0.25, label=LAB[m], zorder=3)
    ax.set_xticks(xg)
    ax.set_xticklabels(
        [f"{PRETTY[f]}\n(n = {rows[f]['n']})" for f in ORDER])
    ax.set_ylabel("Recovery, t = 0.1 mean (%)")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper right", ncol=5, columnspacing=1.0,
              handlelength=1.2, borderaxespad=0.1)
    C.save_si(fig, "family_strata")

if __name__ == "__main__":
    main()
