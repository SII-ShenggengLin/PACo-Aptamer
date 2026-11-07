"""SI Fig. | Confidence diagnostics(置信度–恢复率耦合诊断)。"""
import numpy as np

import si_common as C
from fig_style import panel_label

def _name(m):
    return "PACo-Aptamer" if m == "PACoAptamer" else m

def _load_methods():
    paco_alignif = C.load("paco_confidence.json")
    baselines = C.load("cross_method_confidence.json")
    grnade = C.load("grnade_confidence.json")
    out = {}
    for method in ("PACoAptamer", "AlignIF"):
        record = paco_alignif["methods"][method]
        out[method] = dict(
            per_chain=record["per_chain"], rec_key="recovery_t01_mean",
            rho=record["chain_level_spearman"]["mean_logp_vs_recovery_t01"],
            within=record["within_chain_spearman"],
            enrich=record["top10pct_enrichment"])
    for method in ("RhoDesign", "NA-MPNN"):
        record = baselines["methods"][method]
        summary = baselines["summary_table"]
        out[method] = dict(
            per_chain=record["per_chain"], rec_key="recovery_t01_mean_recomputed",
            rho=summary[method]["chain_level_spearman_t01"],
            within=summary[method]["within_chain_spearman"],
            enrich=summary[method]["top10pct_enrichment"])
    out["gRNAde"] = dict(
        per_chain=grnade["per_chain"], rec_key="recovery_t01_mean_recomputed",
        rho=grnade["chain_level_spearman_t01"],
        within=grnade["within_chain_spearman"],
        enrich=grnade["top10pct_enrichment"])
    return out

def draw_scatter(ax, m, md, letter=None):
    if letter:
        panel_label(ax, letter)
    pc = md["per_chain"]
    x = np.array([v["mean_logp"] for v in pc.values()])
    y = np.array([100 * v[md["rec_key"]] for v in pc.values()])
    ax.scatter(x, y, s=3.2, color=C.MCOLOR[m], lw=0, alpha=0.45)
    r = md["rho"]
    ax.text(0.04, 0.96,
            f"ρ = {r['rho']:.3f}\n95% CI {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}\n"
            f"n = {len(x)}",
            transform=ax.transAxes, fontsize=5.5, va="top", color="black",
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    ax.set_title(_name(m), fontsize=6.0, color="black")
    ax.set_xlabel("Mean log probability\nof sampled designs", fontsize=6.0,
                  color="black")
    ax.set_ylabel("Recovery, t = 0.1 mean (%)" if m == "PACoAptamer" else "",
                  fontsize=6.0, color="black")
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.locator_params(axis="x", nbins=4)
    ax.tick_params(labelsize=5.5, colors="black")

def draw_within(ax, meths):
    panel_label(ax, "b")
    bins = np.linspace(-1, 1, 41)
    notes = []
    for m in meths:
        w = meths[m]["within"]
        if not isinstance(w, dict) or w.get("status", "").startswith("n/a"):
            notes.append(f"{_name(m)}: n/a")
            continue
        pc = meths[m]["per_chain"]
        key = "within_rho"
        vals = [v.get(key) for v in pc.values()]
        vals = np.array([v for v in vals if v is not None], dtype=float)
        if m == "NA-MPNN" and len(vals) == 0:
            vals = np.array([v["within_rho_n10_descriptive"]
                             for v in pc.values()
                             if v.get("within_rho_n10_descriptive") is not None])
            lab = f"NA-MPNN (mean {w.get('descriptive_n10_mean', np.mean(vals)):.3f})"
        else:
            lab = (f"{_name(m)} (mean {w['mean']:.3f}, "
                   f"{100 * w['frac_positive']:.0f}% positive)")
        if m == "PACoAptamer":
            ax.hist(vals, bins=bins, histtype="stepfilled", color=C.MCOLOR[m],
                    alpha=0.55, lw=0.5, edgecolor=C.MCOLOR[m], label=lab)
        else:
            ax.hist(vals, bins=bins, histtype="step", color=C.MCOLOR[m],
                    lw=1.0, label=lab)
    ax.axvline(0, color="#7A7672", lw=0.6)
    ax.set_xlabel("Within-chain Spearman ρ", fontsize=6.0, color="black")
    ax.set_ylabel("Chains", fontsize=6.0, color="black")
    ax.tick_params(labelsize=5.5, colors="black")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 130)
    ax.legend(loc="upper left", fontsize=5.5, borderaxespad=0.1,
              labelspacing=0.28, labelcolor="black")
    if notes:
        ax.text(0.98, 0.97, "\n".join(notes), transform=ax.transAxes,
                fontsize=5.5, ha="right", va="top", color="black")

def draw_enrich(ax, meths):
    panel_label(ax, "c")
    keep = [m for m in meths
            if isinstance(meths[m]["enrich"], dict)
            and isinstance(meths[m]["enrich"].get("mean"), (int, float))]
    x = np.arange(len(keep))
    vals = [100 * meths[m]["enrich"]["mean"] for m in keep]
    lo = [100 * (meths[m]["enrich"]["mean"] - meths[m]["enrich"]["ci95"][0])
          for m in keep]
    hi = [100 * (meths[m]["enrich"]["ci95"][1] - meths[m]["enrich"]["mean"])
          for m in keep]
    ax.bar(x, vals, 0.58, color=[C.MCOLOR[m] for m in keep], ec="white",
           lw=0.3, yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(0, color="#7A7672", lw=0.6)
    for xi, v, h, l in zip(x, vals, hi, lo):
        if v >= 0:
            ax.text(xi, v + h + 0.10, f"{v:+.2f}", ha="center", va="bottom",
                    fontsize=5.5, color="black",
                    bbox=dict(fc="white", ec="none", pad=0.6))
        else:
            ax.text(xi, v - l - 0.10, f"{v:+.2f}", ha="center", va="top",
                    fontsize=5.5, color="black",
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([_name(m) for m in keep], fontsize=5.5, rotation=20,
                       color="black")
    ax.set_ylabel("Top-10% by probability\n− pool mean (pp)", fontsize=6.0,
                  color="black")
    ax.tick_params(labelsize=5.5, colors="black")
    ax.set_ylim(-1.35, 2.3)

def main():
    meths = _load_methods()
    order = [m for m in ("PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF")
             if m in meths]
    nsc = len(order)
    fig, gs = C.make_canvas(96, 2, 1, height_ratios=[1.0, 0.82],
                            hspace=0.50, left=0.065, right=0.995,
                            top=0.90, bottom=0.11)
    gs_top = gs[0].subgridspec(1, 5, wspace=0.42)

    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[3, 2], wspace=0.15)
    for i, m in enumerate(order):
        draw_scatter(fig.add_subplot(gs_top[0, i]), m, meths[m],
                     letter="a" if i == 0 else None)
    if nsc < 4:
        axn = fig.add_subplot(gs_top[0, nsc:])
        axn.axis("off")
        axn.text(0.02, 0.9, "Baseline per-sequence log probabilities\n"
                            "not yet merged", fontsize=5.5, va="top",
                 color="black")
    draw_within(fig.add_subplot(gs_bot[0, 0]), meths)
    draw_enrich(fig.add_subplot(gs_bot[0, 1]), meths)
    C.save_si(fig, "confidence")

if __name__ == "__main__":
    main()
