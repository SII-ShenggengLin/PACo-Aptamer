"""SI Fig. | Dataset composition(数据集描述)。"""
import os

import numpy as np

import si_common as C
from fig_style import panel_label

FINETUNE_CHAINS = 768

SPLITS = os.path.join(C.DATA_DIR, "splits")

def _lengths_by_split():
    seqs = {}
    name, seq = None, []
    with open(os.path.join(SPLITS, "sequences.fasta")) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = len("".join(seq))
                name, seq = line[1:], []
            else:
                seq.append(line)
        if name is not None:
            seqs[name] = len("".join(seq))
    import json
    sp = json.load(open(os.path.join(SPLITS, "splits.json")))
    out = {}
    for k in ["train", "val", "test", "independent_test", "zero_shot_targets"]:
        miss = [s for s in sp[k] if s not in seqs]
        assert not miss, f"missing seqs for {k}: {miss[:5]}"
        out[k] = np.array([seqs[s] for s in sp[k]])
    return sp, out

COL_TRAIN, COL_VAL = "#6B7B8D", "#8FA6BB"
COL_TEST, COL_TIME, COL_ZS = "#B7B0A2", "#A99AAF", "#B5838D"
COL_FT = "#C08B7C"

def draw_a(ax, sp):
    panel_label(ax, "a")
    rows = [("Stage-1 pretraining (train)", len(sp["train"]), COL_TRAIN),
            ("Validation", len(sp["val"]), COL_VAL),
            ("Cluster-based test", len(sp["test"]), COL_TEST),
            ("Time-based test (2024+)", len(sp["independent_test"]), COL_TIME),
            ("Zero-shot test (ligand holdout)", len(sp["zero_shot_targets"]),
             COL_ZS),
            ("Stage-2 finetuning chains", FINETUNE_CHAINS, COL_FT)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, n, col) in zip(y, rows):
        ax.barh(yi, n, 0.62, color=col, ec="white", lw=0.3)
        ax.text(n + 90, yi, f"{n:,}", va="center", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Chains")
    ax.set_xlim(0, 7300)

def draw_b(ax, lens):
    panel_label(ax, "b")
    series = [("train", "Train", COL_TRAIN), ("val", "Validation", COL_VAL),
              ("test", "Cluster-based test", COL_TEST),
              ("independent_test", "Time-based test", COL_TIME),
              ("zero_shot_targets", "Zero-shot test", COL_ZS)]

    assert [int(np.median(lens[k])) for k, _, _ in series] == \
        [71, 58, 43, 37, 45]
    bins = np.arange(0, 420, 10)
    for key, lab, col in series:
        v = lens[key]
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.step(bins[:-1], h * 100, where="post", color=col, lw=1.0,
                label=f"{lab} (n = {len(v):,}, median {int(np.median(v))} nt)")
        ax.fill_between(bins[:-1], h * 100, step="post", color=col, alpha=0.10, lw=0)
    ax.set_xlabel("Chain length (nt)")
    ax.set_ylabel("Density (%)")
    ax.set_xlim(0, 410)
    ax.legend(loc="upper right", fontsize=5.2, borderaxespad=0.1,
              labelspacing=0.3, handlelength=1.4)

PRETTY = {"fam_1MMS": "1MMS", "theophylline_aptamer": "Theophylline",
          "purine_riboswitch": "Purine", "fam_3GCA": "3GCA",
          "dG_riboswitch": "dG", "ztp_riboswitch": "ZTP",
          "mango3_aptamer": "Mango-III", "xanthine_riboswitch": "Xanthine",
          "fam_8HB1": "8HB1", "other": "Other"}

def draw_c(ax):
    panel_label(ax, "c")
    rows = C.load_csv("tables/family_strata.csv")
    csv_n = {r["family"]: int(r["n"]) for r in rows}
    j = C.load("family_stratified_recovery.json")
    for fam, g in j["groups"].items():
        assert g["n_chains"] == csv_n[fam], f"n mismatch {fam}"
    assert j["n_other"] == csv_n["other"] == 372
    assert sum(csv_n.values()) == 477
    fams = sorted(csv_n, key=lambda f: -csv_n[f])
    fams = [f for f in fams if f != "other"] + ["other"]
    y = np.arange(len(fams))[::-1]
    for yi, f in zip(y, fams):
        col = "#B7B0A2" if f == "other" else "#D9C3A9"
        ax.barh(yi, csv_n[f], 0.62, color=col, ec="white", lw=0.3)
        ax.text(csv_n[f] + 8, yi, str(csv_n[f]), va="center", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[f] for f in fams])
    ax.set_xlabel("Chains in cluster-based test")
    ax.set_xlim(0, 440)

def main():
    sp, lens = _lengths_by_split()
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"]),
            len(sp["independent_test"]), len(sp["zero_shot_targets"])) == \
        (5773, 96, 477, 235, 93)
    fig, gs = C.make_canvas(58, 1, 3, width_ratios=[1.0, 1.2, 1.0],
                            wspace=0.38, left=0.155, right=0.99,
                            top=0.88, bottom=0.20)
    draw_a(fig.add_subplot(gs[0, 0]), sp)
    draw_b(fig.add_subplot(gs[0, 1]), lens)
    draw_c(fig.add_subplot(gs[0, 2]))
    C.save_si(fig, "dataset")

if __name__ == "__main__":
    main()
