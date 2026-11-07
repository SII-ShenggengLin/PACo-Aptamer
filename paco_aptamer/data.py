import random
from pathlib import Path

import numpy as np
import torch

G27 = [
    "P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
    "N6", "N7", "C8", "N9", "OP1", "OP2",
]
PACO9 = ["C4'", "C1'", "N1", "C2", "C5'", "O5'", "P", "C2'", "C3'"]
PACO9_IDX = [G27.index(atom) for atom in PACO9]
ELEMENT_ORDER = ["C", "N", "O", "P", "S"]
FILL = 1e-5


def to_paco9(coords_27):
    coords = coords_27[:, PACO9_IDX, :].astype(np.float32).copy()
    fill_rows = np.abs(coords - FILL).max(axis=-1) < 1e-4
    coords[fill_rows] = np.nan
    return coords


def ss_to_pairs(secondary_structure):
    length = len(secondary_structure)
    pairs = np.zeros((length, length), dtype=bool)
    stacks = {opening: [] for opening in "([{<"}
    matching = dict(zip(")]}>", "([{<"))
    for i, char in enumerate(secondary_structure):
        if char in stacks:
            stacks[char].append(i)
        elif char in matching:
            stack = stacks[matching[char]]
            if stack:
                j = stack.pop()
                pairs[i, j] = pairs[j, i] = True
    return pairs


def element_onehot(element):
    vector = np.zeros(5, dtype=np.float32)
    if element in ELEMENT_ORDER:
        vector[ELEMENT_ORDER.index(element)] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def lig_features(ligand_coords, ligand_elements, rna_c1p):
    if len(ligand_coords) == 0:
        return np.zeros((0, 21), dtype=np.float32)
    features = np.stack([element_onehot(element) for element in ligand_elements])
    distances = np.linalg.norm(
        ligand_coords[:, None, :] - rna_c1p[None, :, :], axis=-1
    ).min(axis=-1)
    distances = np.clip(distances, 0.0, 20.0)
    centers = np.linspace(0.0, 20.0, 16)
    sigma = 20.0 / 16
    rbf = np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * sigma**2))
    return np.concatenate([features, rbf], axis=-1).astype(np.float32)


def load_rna_features(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_complex_features(path):
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    output = {}
    for stem, sample in raw.items():
        coords27 = np.asarray(sample["coords_27"], dtype=np.float32)
        sequence = sample["sequence"]
        secondary_structure = sample["ss"]
        if coords27.shape[0] != len(sequence) or len(secondary_structure) != len(sequence):
            continue
        rna_c1p = coords27[:, G27.index("C1'"), :]
        features = lig_features(
            np.asarray(sample["ligand_coords"], dtype=np.float32),
            sample["ligand_elements"],
            rna_c1p,
        )
        if features.shape[0] == 0:
            continue
        output[stem] = {
            "coords9": to_paco9(coords27),
            "ss_pairs": ss_to_pairs(secondary_structure),
            "seq": sequence,
            "lig_feats": features,
        }
    return output


def augment(coords9, noise_std=0.1):
    coords = coords9.copy()
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(0, np.pi)
    psi = random.uniform(0, 2 * np.pi)
    rz1 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)],
    ])
    rz2 = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1],
    ])
    coords = coords @ (rz2 @ rx @ rz1).T
    valid = np.all(np.isfinite(coords), axis=-1)
    noise = np.random.normal(0, noise_std, coords.shape).astype(np.float32)
    return np.where(valid[..., None], coords + noise, coords).astype(np.float32)
