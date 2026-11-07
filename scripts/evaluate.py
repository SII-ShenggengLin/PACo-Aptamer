#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paco_aptamer.data import load_rna_features  # noqa: E402
from paco_aptamer.inference import (  # noqa: E402
    DEFAULT_TEMPERATURES,
    MAX_SEQUENCE_LENGTH,
    load_ensemble,
    resolve_device,
    sample_chain,
    sequence_recovery,
)

DEFAULT_CHECKPOINTS = [
    ROOT / "checkpoints/paco_aptamer/ensemble_member_1.pth",
    ROOT / "checkpoints/paco_aptamer/ensemble_member_2.pth",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PACo-Aptamer on a released split.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "independent_test", "zero_shot_targets"],
        default="test",
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--samples-per-temperature", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device, for example cpu, cuda or cuda:0.")
    parser.add_argument("--max-chains", type=int, help="Evaluate only the first N chains.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits_path = args.data_dir / "splits/splits.json"
    features_path = args.data_dir / "features/rna.pt"
    splits = json.loads(splits_path.read_text())
    features = load_rna_features(features_path)
    stems = [stem for stem in splits[args.split] if stem in features]
    if args.max_chains is not None:
        stems = stems[: args.max_chains]
    if not stems:
        raise ValueError("No selected chains have released features.")
    if args.temperatures[0] != 0.1:
        raise ValueError("The first temperature must be 0.1 for the reported low-temperature metric.")
    device = resolve_device(args.device)
    models, alphabet = load_ensemble(args.checkpoints, device)

    per_chain = {}
    started = time.time()
    for i, stem in enumerate(stems, 1):
        feature = features[stem]
        if len(feature["seq"]) > MAX_SEQUENCE_LENGTH:
            continue
        native = feature["seq"]
        pool = []
        for temperature in args.temperatures:
            pool.extend(sample_chain(
                models,
                alphabet,
                feature,
                device,
                temperature,
                args.samples_per_temperature,
            ))
        recoveries = [sequence_recovery(sequence, native) for sequence in pool]
        low_temperature_count = args.samples_per_temperature
        per_chain[stem] = {
            "len": len(native),
            "recovery_best": max(recoveries),
            "recovery_t01_mean": float(np.mean(recoveries[:low_temperature_count])),
            "n_pool": len(pool),
        }
        if i % 25 == 0 or i == len(stems):
            print(f"Evaluated {i}/{len(stems)} chains in {time.time() - started:.0f}s", flush=True)
    if not per_chain:
        raise ValueError("No chains satisfied the released length limit.")
    summary = {
        "split": args.split,
        "n_chains": len(per_chain),
        "recovery_best_mean": float(np.mean([
            record["recovery_best"] for record in per_chain.values()
        ])),
        "recovery_best_median": float(np.median([
            record["recovery_best"] for record in per_chain.values()
        ])),
        "recovery_t01_mean_mean": float(np.mean([
            record["recovery_t01_mean"] for record in per_chain.values()
        ])),
        "temperatures": args.temperatures,
        "samples_per_temperature": args.samples_per_temperature,
        "seed": args.seed,
    }
    output = args.output or ROOT / f"outputs/benchmark_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "per_chain": per_chain}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
