#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paco_aptamer.inference import (  # noqa: E402
    DEFAULT_TEMPERATURES,
    MAX_SEQUENCE_LENGTH,
    load_ensemble,
    resolve_device,
    sample_chain,
)
from paco_aptamer.structure import feature_from_pdb  # noqa: E402

DEFAULT_CHECKPOINTS = [
    ROOT / "checkpoints/paco_aptamer/ensemble_member_1.pth",
    ROOT / "checkpoints/paco_aptamer/ensemble_member_2.pth",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Design RNA sequences for a PDB backbone with PACo-Aptamer."
    )
    parser.add_argument("--pdb", type=Path, required=True, help="Input RNA PDB file.")
    parser.add_argument("--chain", help="RNA chain identifier; required for multi-chain files.")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="One or more checkpoints; multiple checkpoints use probability averaging.",
    )
    parser.add_argument("--output", type=Path, default=Path("paco_designs.fasta"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--samples-per-temperature", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device, for example cpu, cuda or cuda:0.")
    parser.add_argument("--rnafold", default="RNAfold", help="RNAfold executable.")
    parser.add_argument(
        "--secondary-structure",
        help="Optional dot-bracket string; skips RNAfold when supplied.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(missing) +
            ". Download and unpack the Zenodo data package in the repository root."
        )
    if args.samples_per_temperature < 1 or any(t <= 0 for t in args.temperatures):
        raise ValueError("Temperatures and sample counts must be positive.")
    feature = feature_from_pdb(
        args.pdb,
        chain=args.chain,
        rnafold=args.rnafold,
        secondary_structure=args.secondary_structure,
    )
    length = len(feature["seq"])
    if length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"The released model supports at most {MAX_SEQUENCE_LENGTH} nucleotides; got {length}."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    models, alphabet = load_ensemble(args.checkpoints, device)

    designs = []
    for temperature in args.temperatures:
        sequences = sample_chain(
            models,
            alphabet,
            feature,
            device,
            temperature,
            args.samples_per_temperature,
        )
        designs.extend((temperature, sequence) for sequence in sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for i, (temperature, sequence) in enumerate(designs, 1):
            handle.write(f">design_{i:04d} temperature={temperature}\n{sequence}\n")
    print(f"Input: {args.pdb} ({length} nt)")
    print(f"Device: {device}; checkpoints: {len(models)}; designs: {len(designs)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
