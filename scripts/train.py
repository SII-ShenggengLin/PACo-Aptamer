#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paco_aptamer.alphabet import Alphabet  # noqa: E402
from paco_aptamer.data import augment, load_complex_features, load_rna_features  # noqa: E402
from paco_aptamer.inference import (  # noqa: E402
    MAX_SEQUENCE_LENGTH,
    default_model_args,
    resolve_device,
)
from paco_aptamer.model import PACoModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the released PACo-Aptamer configuration.")
    parser.add_argument("--stage", choices=["s1", "s2"], required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/checkpoints")
    parser.add_argument("--name", help="Checkpoint name without the .pth suffix.")
    parser.add_argument("--init", type=Path, help="Stage-1 checkpoint used to initialize stage 2.")
    parser.add_argument("--epochs", type=int, help="Defaults to 60 for stage 1 and 8 for stage 2.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-chains", type=int, default=4)
    parser.add_argument("--ss-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", help="PyTorch device, for example cpu, cuda or cuda:0.")
    return parser.parse_args()


def train_epoch(
    model,
    data,
    stems,
    alphabet,
    optimizer,
    device,
    batch_chains,
    ss_dropout,
    rna_data=None,
    rna_stems=None,
):
    model.train()
    padding_token = alphabet.padding_idx
    bos = alphabet.get_idx("<cath>")
    entries = [(stem, data, True) for stem in stems]
    if rna_data is not None:
        selected = list(rna_stems)
        random.shuffle(selected)
        entries.extend((stem, rna_data, False) for stem in selected[: len(stems)])
    entries = [
        entry for entry in entries
        if len(entry[1][entry[0]]["seq"]) <= MAX_SEQUENCE_LENGTH
    ]
    entries.sort(key=lambda entry: len(entry[1][entry[0]]["seq"]))
    batches = [entries[i : i + batch_chains] for i in range(0, len(entries), batch_chains)]
    random.shuffle(batches)
    total_loss = 0.0
    total_tokens = 0

    for batch in batches:
        batch_size = len(batch)
        max_length = max(len(source[stem]["seq"]) for stem, source, _ in batch)
        ligand_sizes = [
            source[stem].get("lig_feats", np.zeros((0, 21))).shape[0]
            if has_ligand else 0
            for stem, source, has_ligand in batch
        ]
        max_ligands = max(ligand_sizes, default=0)
        coords = torch.zeros(
            batch_size, max_length, 9, 3, dtype=torch.float32, device=device
        )
        secondary_structure = torch.zeros(
            batch_size, max_length, max_length, dtype=torch.bool, device=device
        )
        tokens = torch.full(
            (batch_size, max_length + 1), padding_token, dtype=torch.long, device=device
        )
        padding_mask = torch.ones(batch_size, max_length, dtype=torch.bool, device=device)
        confidence = torch.ones(batch_size, max_length, device=device)
        ligand_features = torch.zeros(batch_size, max_ligands, 21, device=device)
        ligand_mask = torch.zeros(batch_size, max_ligands, dtype=torch.bool, device=device)
        tokens[:, 0] = bos

        for j, (stem, source, has_ligand) in enumerate(batch):
            sample = source[stem]
            length = len(sample["seq"])
            augmented = augment(sample["coords9"])
            coords[j, :length] = torch.as_tensor(
                augmented, dtype=torch.float32, device=device
            )
            coords[j, length:] = torch.as_tensor(
                augmented[0:1], dtype=torch.float32, device=device
            )
            if not (ss_dropout > 0 and random.random() < ss_dropout):
                secondary_structure[j, :length, :length] = torch.as_tensor(
                    sample["ss_pairs"], dtype=torch.bool, device=device
                )
            tokens[j, 1 : length + 1] = torch.as_tensor(
                [alphabet.get_idx(char) for char in sample["seq"]],
                dtype=torch.long,
                device=device,
            )
            padding_mask[j, :length] = False
            if has_ligand and ligand_sizes[j] > 0:
                count = ligand_sizes[j]
                ligand_features[j, :count] = torch.as_tensor(
                    sample["lig_feats"], dtype=torch.float32, device=device
                )
                ligand_mask[j, :count] = True

        logits = model(
            coords,
            secondary_structure,
            ligand_features,
            ligand_mask,
            padding_mask,
            confidence,
            prev_output_tokens=tokens[:, :-1],
        ).permute(0, 2, 1)
        target = tokens[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=padding_token,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = (target != padding_token).sum().item()
        total_loss += loss.item() * count
        total_tokens += count
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_validation(model, data, stems, alphabet, device):
    model.eval()
    recoveries = []
    for stem in stems:
        sample = data[stem]
        length = len(sample["seq"])
        if length > MAX_SEQUENCE_LENGTH:
            continue
        coords = torch.as_tensor(
            sample["coords9"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        secondary_structure = torch.as_tensor(
            sample["ss_pairs"], dtype=torch.bool, device=device
        ).unsqueeze(0)
        confidence = torch.ones(1, length, device=device)
        ligand = torch.zeros(1, 0, 21, device=device)
        ligand_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        prediction = model.sample(
            coords,
            secondary_structure,
            ligand,
            ligand_mask,
            confidence,
            device,
            temperature=0.1,
        )
        recovery = sum(
            alphabet.get_idx(native) == predicted
            for native, predicted in zip(sample["seq"], prediction.tolist())
        ) / max(1, length)
        recoveries.append(recovery)
    return float(np.mean(recoveries)) if recoveries else 0.0


def main():
    args = parse_args()
    if args.batch_chains < 1 or not 0 <= args.ss_dropout <= 1:
        raise ValueError("Batch size must be positive and SS dropout must be in [0, 1].")
    if args.stage == "s2" and args.init is None:
        raise ValueError("Stage 2 requires --init /path/to/stage1.pth.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.stage
    output_path = args.output_dir / f"{name.removesuffix('.pth')}.pth"

    splits = json.loads((args.data_dir / "splits/splits.json").read_text())
    train_stems = list(splits["train"])
    val_stems = list(splits["val"])
    rna = load_rna_features(args.data_dir / "features/rna.pt")
    train_stems = [stem for stem in train_stems if stem in rna]
    val_stems = [stem for stem in val_stems if stem in rna]
    data = rna
    if args.stage == "s2":
        data = load_complex_features(args.data_dir / "features/complexes_raw.pt")
        train_stems = list(data)
        s2_pdbs = {stem.split("_")[0] for stem in data}
        val_stems = [stem for stem in val_stems if stem.split("_")[0] not in s2_pdbs]
        print(f"Stage-2 chains: {len(train_stems)}; clean validation chains: {len(val_stems)}")

    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = PACoModel(default_model_args(), alphabet).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    epochs = args.epochs or (60 if args.stage == "s1" else 8)
    best_recovery = -1.0
    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = train_epoch(
            model,
            data,
            train_stems,
            alphabet,
            optimizer,
            device,
            args.batch_chains,
            args.ss_dropout,
            rna_data=rna if args.stage == "s2" else None,
            rna_stems=[stem for stem in splits["train"] if stem in rna]
            if args.stage == "s2" else None,
        )
        should_evaluate = args.stage == "s2" or epoch % 5 == 0 or epoch == epochs
        if should_evaluate:
            recovery = evaluate_validation(model, rna, val_stems, alphabet, device)
            print(
                f"{args.stage} epoch {epoch}: loss={loss:.4f} "
                f"val_recovery={recovery:.4f} time={time.time() - started:.0f}s",
                flush=True,
            )
            if recovery > best_recovery:
                best_recovery = recovery
                torch.save(model.state_dict(), output_path)
                print(f"Saved {output_path}", flush=True)
        else:
            print(
                f"{args.stage} epoch {epoch}: loss={loss:.4f} "
                f"time={time.time() - started:.0f}s",
                flush=True,
            )
    print(f"Best validation recovery: {best_recovery:.4f}")


if __name__ == "__main__":
    main()
