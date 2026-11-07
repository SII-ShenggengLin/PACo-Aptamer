"""SI Fig. | Fingerprint robustness(Tanimoto 稳健性三指纹)。"""
import numpy as np
from scipy import stats

import si_common as C
from fig_style import panel_label

FPS = [("morgan", "Morgan (radius 2, 2048 bit)"),
       ("maccs", "MACCS keys (167 bit)"),
       ("rdkit_topo", "RDKit topological (2048 bit)")]
EXPECT = {"morgan": (-0.4007, 1.203e-04), "maccs": (-0.3406, 1.246e-03),
          "rdkit_topo": (-0.3448, 1.074e-03)}

def main():
    d = C.load("ligand_fingerprint_robustness.json")
    t93 = C.load("ligand_novelty.json")
    s93 = {s["stem"]: s["max_tanimoto"] for s in t93["scatter"]}
    pc = d["per_chain"]

    for stem, v in pc.items():
        assert abs(v["max_tanimoto_morgan"] - s93[stem]) < 1e-12, stem
    for fp, (er, ep) in EXPECT.items():
        sp = d["spearman"][fp]["sim_vs_delta_t01"]
        assert abs(sp["rho"] - er) < 5e-4 and abs(sp["p"] - ep) < 1e-5, fp

    fig, gs = C.make_canvas(58, 1, 3, wspace=0.34, left=0.065, right=0.995,
                            top=0.88, bottom=0.18)
    for i, (fp, lab) in enumerate(FPS):
        ax = fig.add_subplot(gs[0, i])
        panel_label(ax, "abc"[i])
        x = np.array([v[f"max_tanimoto_{fp}"] for v in pc.values()])
        y = np.array([100 * v["delta_t01_PACo_minus_RhoDesign"]
                      for v in pc.values()])
        ax.scatter(x, y, s=5.0, color=C.PACO, lw=0, alpha=0.75, zorder=3)
        lr = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, lr.slope * xs + lr.intercept, ls="--", lw=0.9,
                color="#7A7672", zorder=2)
        ax.axhline(0, color="#7A7672", lw=0.5, zorder=1)
        sp = d["spearman"][fp]["sim_vs_delta_t01"]
        ax.text(0.03, 0.04,
                f"Spearman ρ = {sp['rho']:.3f}\n{C.pfmt_sci(sp['p'])}, "
                f"n = {len(x)}",
                transform=ax.transAxes, fontsize=5.2, va="bottom",
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))
        ax.set_title(lab)
        ax.set_xlabel("Maximum Tanimoto similarity")
        if i == 0:
            ax.set_ylabel("Δ recovery, t = 0.1 mean (%)")
        ax.set_ylim(-14, 24)
    C.save_si(fig, "fingerprint")

if __name__ == "__main__":
    main()
