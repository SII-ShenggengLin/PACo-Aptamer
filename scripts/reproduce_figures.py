#!/usr/bin/env python3
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MAIN_FIGURES = [f"fig{i}" for i in range(1, 7)]
SUPPLEMENTARY_FIGURES = [
    "si_confidence",
    "si_dataset",
    "si_efficiency",
    "si_family_strata",
    "si_fingerprint",
    "si_funnel",
    "si_iface_logo",
    "si_perplexity",
    "si_predbb",
    "si_scmcc_sctm",
    "si_ss_attention",
    "si_temperature",
]
EXTENDED_DATA_FIGURES = [f"ed_fig{i}" for i in range(1, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate figures from released final results.")
    parser.add_argument(
        "figures",
        nargs="*",
        help="Figure names, for example fig2 ed_fig1 si_dataset; default is all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("PACO_DATA_DIR", ROOT / "data")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested = args.figures or MAIN_FIGURES + EXTENDED_DATA_FIGURES + SUPPLEMENTARY_FIGURES
    known = set(MAIN_FIGURES + EXTENDED_DATA_FIGURES + SUPPLEMENTARY_FIGURES)
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown figure name(s): {', '.join(unknown)}")
    if not (args.data_dir / "results").is_dir():
        raise FileNotFoundError(
            f"No released results found under {args.data_dir}. Unpack the Zenodo package first."
        )
    os.environ["PACO_DATA_DIR"] = str(args.data_dir.resolve())
    for name in requested:
        if name in MAIN_FIGURES:
            path = FIGURES / f"{name}.py"
        elif name in EXTENDED_DATA_FIGURES:
            path = FIGURES / "ed" / f"{name}.py"
        else:
            path = FIGURES / "si" / f"{name}.py"
        print(f"Generating {name}", flush=True)
        sys.path.insert(0, str(path.parent))
        try:
            runpy.run_path(str(path), run_name="__main__")
        finally:
            sys.path.remove(str(path.parent))


if __name__ == "__main__":
    main()
