"""SI Fig. | Native-sequence perplexity(逐链 native perplexity)。"""
import glob
import math
import os

import numpy as np

import si_common as C
from fig_style import panel_label

def _paco_ppl():
    out = {}
    for f in glob.glob(os.path.join(C.SUR, "paco_cluster_pools", "*.json")):
        stem = os.path.basename(f)[:-5]
        import json
        out[stem] = math.exp(-json.load(open(f))["logp_native"])
    return out

def main():
    d = C.load("native_perplexity.json")
    paco = _paco_ppl()
    ppl = {"PACoAptamer": paco,
           "RhoDesign": {k: v["ppl_native"] for k, v in d["RhoDesign"].items()},
           "gRNAde": {k: v["ppl_native"] for k, v in d["gRNAde"].items()},
           "NA-MPNN": {k: v["ppl_native"] for k, v in d["NA-MPNN"].items()}}

    assert abs(np.mean(list(ppl["PACoAptamer"].values())) - 2.911) < 5e-4
    for m, ref in [("RhoDesign", 3.450), ("gRNAde", 3.016), ("NA-MPNN", 3.401)]:
        assert abs(d["macro_summary"][m]["ppl_native_macro_mean"] - ref) < 5e-4
        assert abs(np.mean(list(ppl[m].values())) -
                   d["macro_summary"][m]["ppl_native_macro_mean"]) < 1e-9

    order = ["PACoAptamer", "gRNAde", "NA-MPNN", "RhoDesign"]
    fig, gs = C.make_canvas(64, 1, 2, width_ratios=[1.15, 1.0],
                            wspace=0.30, left=0.075, right=0.99,
                            top=0.88, bottom=0.17)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    ax.text(0.03, 0.97, "horizontal bar = macro mean (n = 477 chains)",
            transform=ax.transAxes, fontsize=4.8, va="top", color="black")
    rng = np.random.default_rng(7)
    for i, m in enumerate(order):
        v = np.array(list(ppl[m].values()))
        vp = ax.violinplot(v, positions=[i], widths=0.62, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(C.MCOLOR[m]); b.set_alpha(0.45); b.set_lw(0)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=1.6,
                   color=C.MCOLOR[m], alpha=0.25, lw=0)
        ax.plot([i - 0.26, i + 0.26], [v.mean(), v.mean()], color="black",
                lw=1.0, zorder=5)
        ax.text(i, v.mean() + 0.35, f"{v.mean():.3f}", ha="center",
                fontsize=5.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([C.MLAB[m] for m in order], fontsize=5.6)
    ax.set_ylabel("Perplexity of the native sequence")
    ax.set_ylim(1.2, 15.5)
    ax.set_yscale("log")
    ax.set_yticks([1.5, 2, 3, 4, 5, 7, 10, 15])
    ax.set_yticklabels(["1.5", "2", "3", "4", "5", "7", "10", "15"])
    ax.minorticks_off()

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    bins = np.linspace(-3, 8, 60)
    for m in ["gRNAde", "NA-MPNN", "RhoDesign"]:
        common = sorted(set(ppl[m]) & set(ppl["PACoAptamer"]))
        delta = np.array([ppl[m][k] - ppl["PACoAptamer"][k] for k in common])
        assert len(common) == 477
        ax.hist(delta, bins=bins, histtype="step", color=C.MCOLOR[m], lw=1.0,
                label=f"{C.MLAB[m]} − PACo-Aptamer (mean Δ {delta.mean():+.3f})")
    ax.axvline(0, color="#7A7672", lw=0.6)
    ax.set_xlabel("Per-chain Δ perplexity (baseline − PACo-Aptamer)")
    ax.set_ylabel("Chains")
    ax.legend(loc="upper right", fontsize=5.0, borderaxespad=0.3)
    C.save_si(fig, "perplexity")

if __name__ == "__main__":
    main()
