#!/usr/bin/env python3
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
NAME = "PACoAptamer_research_data_v1"
DST = HERE / NAME
TARBALL = HERE / f"{NAME}.tar.gz"

DIRECTORIES = [
    "data/splits_v2",
    "data/canonical_v2",
    "data/chain_pdb_v2",
    "data/ccd",
    "data/alignif_btrain_v2",
    "data/grnade_btrain_v2",
    "data/nampnn_btrain_v2",
    "data/survey/100_predbb_inputs",
    "data/survey/101_predbb_inputs",
    "data/survey/102_predbb_inputs",
    "data/survey/104_ed_tables",
    "data/survey/109_si_tables",
    "data/survey/eval_inputs_v2",
    "data/survey/grnade_inputs_v2",
    "data/survey/grnade_inputs_btest_v2",
    "data/survey/grnade_inputs_predbb",
    "data/survey/grnade_out_v2",
    "data/survey/77_ens2_pool_btest",
    "data/survey/paco_fix_pool_ens2noiface",
    "data/survey/paco_av2_pool_ens2noiface",
    "data/survey/rhodesign_out_btest_v2",
    "data/survey/grnade_out_btest_v2_cfix",
    "data/survey/alignif_out_btest_v2",
    "data/survey/nampnn_out_btest_v2",
    "data/survey/rhodesign_out_av2_retrained",
    "data/survey/grnade_out_av2_retrained_cfix",
    "data/survey/alignif_out_av2_retrained",
    "data/survey/nampnn_out_av2_retrained",
    "data/survey/94_predbb_pool",
    "data/survey/94_predbb_pool_crystal",
    "data/survey/95_pools",
    "data/survey/74_pools",
    "data/survey/75_pools",
    "data/survey/75_score",
    "data/survey/86_sctm",
    "data/survey/97_ss_robust_chains",
    "data/survey/79_rhofold",
    "data/survey/88_rhofold",
]

FILES = [
    "data/paco_s1_v2.pt",
    "data/survey/75_feats.pt",
    "models/PACoAptamer/s2_v2_seed43_noiface.pth",
    "models/PACoAptamer/s2_v2_ssdo15_noiface.pth",
    "models/RhoDesign_btrain_v2/best_val.pth",
    "models/gRNAde_btrain_v2/best_checkpoint.h5",
    "models/NA-MPNN_btrain_v2/s_11016.pt",
    "models/AlignIF_btrain_v2/alignif-seed=0-epoch=79.ckpt",
]

README = """# PACo-Aptamer research data (v1.0.0)

Research data supporting **PACo-Aptamer: ligand-conditioned inverse folding for RNA aptamer design**.

- Code: https://github.com/SII-ShenggengLin/RNA_tertiary_stru_IF (MIT)
- Data and selected model checkpoints: CC BY 4.0
- Creator: Shenggeng Lin

## Scope

The archive preserves the repository's relative paths. Copy or merge `data/` and
`models/` into a clone of the source repository. The package contains:

1. Final sequence/structure cluster definitions and the train, validation,
   cluster-based test, time-based independent test and ligand-holdout splits.
   Their respective chain counts are 5,773, 96, 477, 235 and 93. The latter two
   test layers overlap the cluster-based test where the same chain satisfies
   multiple holdout criteria; all are disjoint from training.
2. All 6,346 canonical cleaned RNA-chain PDB files used by the final split,
   the repaired complex-chain PDB files, CCD snapshots, and data-quality records.
3. PACo-Aptamer stage-1 features and final stage-2/leave-one-family-out feature
   files, plus the preprocessed data used to retrain AlignIF, gRNAde and NA-MPNN.
4. Final benchmark result JSON files, final designed-sequence pools, interface,
   ablation, predicted-backbone, ligand-similarity, confidence, diversity,
   secondary-structure, self-consistency, efficiency and aptamer-interaction
   analyses, and the machine-readable Extended Data/Supplementary tables.
5. Selected final checkpoints: the two PACo-Aptamer ensemble members and the
   selected retrained RhoDesign, gRNAde, NA-MPNN and AlignIF models.

The PACo-Aptamer ensemble members are
`models/PACoAptamer/s2_v2_seed43_noiface.pth` and
`models/PACoAptamer/s2_v2_ssdo15_noiface.pth`. During benchmark inference the
ligand input is empty; ligand context is used during stage-2 finetuning.

## Integrity check

From the unpacked archive directory run:

```bash
sha256sum -c MANIFEST.sha256
```

The manifest covers every packaged file except itself. The archive has one
top-level directory, `PACoAptamer_research_data_v1/`.

## Exclusions and external requirements

This is a curated reproduction package, not a byte copy of the server's 129 GB
data directory. It excludes the bulk raw RCSB PDB/mmCIF download, regenerable
caches, logs, superseded backups, old development pools, third-party Python
libraries and non-selected checkpoints. Raw structures can be obtained from
RCSB using the identifiers in the splits and metadata. Full retraining and
refolding also require the external software and upstream weights documented in
the GitHub repository, including RhoFold+ for structure prediction.

Serialized PyTorch files are data/model artifacts, not portable executables.
Load them only in an appropriate environment and use the source repository's
versions and protocols.

## Primary-source attribution

Structures and chemical-component annotations derive from the RCSB Protein Data
Bank (https://www.rcsb.org/) and the PDB Chemical Component Dictionary
(https://www.wwpdb.org/data/ccd). Aptamer–target interaction data derive from
AptaDB (https://lmmd.ecust.edu.cn/aptadb/) and the UTexas Aptamer Database
(https://doi.org/10.5281/zenodo.8387047). AlignIF source data are described at
https://doi.org/10.5281/zenodo.20528802. Please cite the original resources as
well as PACo-Aptamer when reusing the corresponding subsets.
"""


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    selected = {Path(p) for p in FILES}
    selected.update(Path("data/s2_btrain") / p.name for p in (ROOT / "data/s2_btrain").glob("s2_samples_v2*.pt") if "smoke" not in p.name)
    for directory in DIRECTORIES:
        base = ROOT / directory
        if not base.is_dir():
            raise FileNotFoundError(base)
        selected.update(p.relative_to(ROOT) for p in base.rglob("*") if p.is_file())
    survey = ROOT / "data/survey"
    for p in survey.iterdir():
        if not p.is_file() or p.suffix not in {".json", ".fa", ".clstr"}:
            continue
        if ".bak" in p.name or "backup" in p.name or "_withiface_" in p.name:
            continue
        selected.add(p.relative_to(ROOT))
    return sorted(selected, key=str)


def main():
    src = sources()
    missing = [str(p) for p in src if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for i, rel in enumerate(src, 1):
        target = DST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
        if i % 5000 == 0:
            print(f"copied {i}/{len(src)}", flush=True)
    (DST / "README.md").write_text(README)
    packaged = sorted((p for p in DST.rglob("*") if p.is_file()), key=str)
    with (DST / "MANIFEST.sha256").open("w") as f:
        for p in packaged:
            f.write(f"{sha256(p)}  {p.relative_to(DST)}\n")
    splits = json.loads((DST / "data/splits_v2/splits_v2.json").read_text())
    assert {k: len(splits[k]) for k in ("train", "val", "test", "independent_test", "zero_shot_targets")} == {
        "train": 5773, "val": 96, "test": 477, "independent_test": 235, "zero_shot_targets": 93}
    assert len(list((DST / "data/canonical_v2").glob("*.pdb"))) == 6346
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(DST, arcname=NAME, recursive=True)
    print(json.dumps({
        "files_including_manifest": len(packaged) + 1,
        "uncompressed_bytes": sum(p.stat().st_size for p in packaged),
        "tarball": str(TARBALL),
        "tarball_bytes": TARBALL.stat().st_size,
        "tarball_sha256": sha256(TARBALL),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
