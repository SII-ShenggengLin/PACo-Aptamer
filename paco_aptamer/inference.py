import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .alphabet import Alphabet
from .model import PACoModel

DEFAULT_TEMPERATURES = (0.1, 0.5, 1.0)
MAX_SEQUENCE_LENGTH = 512


def default_model_args():
    return argparse.Namespace(
        local_rank=-1,
        device_id=[0],
        encoder_embed_dim=512,
        decoder_embed_dim=512,
        dropout=0.1,
        gvp_top_k_neighbors=15,
        gvp_node_hidden_dim_vector=256,
        gvp_node_hidden_dim_scalar=512,
        gvp_edge_hidden_dim_scalar=32,
        gvp_edge_hidden_dim_vector=1,
        gvp_num_encoder_layers=3,
        gvp_dropout=0.1,
        encoder_layers=3,
        encoder_attention_heads=4,
        attention_dropout=0.1,
        encoder_ffn_embed_dim=512,
        decoder_layers=3,
        decoder_attention_heads=4,
        decoder_ffn_embed_dim=512,
    )


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint, device=None):
    device = resolve_device(device)
    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = PACoModel(default_model_args(), alphabet).to(device)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, alphabet


def load_ensemble(checkpoints, device=None):
    device = resolve_device(device)
    models = []
    alphabet = None
    for checkpoint in checkpoints:
        model, alphabet = load_model(checkpoint, device)
        models.append(model)
    if not models:
        raise ValueError("At least one checkpoint is required.")
    return models, alphabet


def model_inputs(feature, device):
    length = len(feature["seq"])
    coords = torch.as_tensor(
        feature["coords9"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    secondary_structure = torch.as_tensor(
        feature["ss_pairs"], dtype=torch.bool, device=device
    ).unsqueeze(0)
    confidence = torch.ones(1, length, device=device)
    ligand = torch.zeros(1, 0, 21, device=device)
    ligand_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
    return coords, secondary_structure, ligand, ligand_mask, confidence


@torch.no_grad()
def sample_chain(models, alphabet, feature, device, temperature, n_samples):
    device = resolve_device(device)
    coords, ss, ligand, ligand_mask, confidence = model_inputs(feature, device)
    if len(models) == 1:
        prediction = models[0].sample(
            coords,
            ss,
            ligand,
            ligand_mask,
            confidence,
            device,
            temperature=temperature,
            n_samples=n_samples,
        )
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(0)
        return [
            "".join(alphabet.get_tok(token) for token in row.tolist())
            for row in prediction
        ]

    sample_count = int(n_samples)
    length = coords.shape[1]
    encoder_outputs = []
    for model in models:
        padding_mask = torch.zeros(1, length, dtype=torch.bool, device=device)
        output = model.forward_embedding(
            coords, ss, ligand, ligand_mask, padding_mask, confidence
        )
        encoder_outputs.append({
            "encoder_out": [output["encoder_out"][0].repeat(1, sample_count, 1)],
            "encoder_padding_mask": [
                output["encoder_padding_mask"][0].repeat(sample_count, 1)
            ],
            "encoder_embedding": [],
            "encoder_states": [],
        })

    prediction = torch.zeros(
        sample_count, length + 1, dtype=torch.long, device=device
    )
    prediction[:, 0] = alphabet.get_idx("<cath>")
    states = [{} for _ in models]
    for i in range(length):
        previous = prediction[:, : i + 1]
        probabilities = None
        for model, encoder_output, state in zip(models, encoder_outputs, states):
            logits, _ = model.decoder(
                previous, encoder_out=encoder_output, incremental_state=state
            )
            current = F.softmax(logits[:, :, -1] / temperature, dim=-1)
            probabilities = current if probabilities is None else probabilities + current
        sampled = torch.multinomial(probabilities / len(models), 1).squeeze(-1)
        prediction[:, i + 1] = sampled
    return [
        "".join(alphabet.get_tok(token) for token in row.tolist())
        for row in prediction[:, 1:]
    ]


def sequence_recovery(designed, native):
    positions = [i for i, nucleotide in enumerate(native) if nucleotide != "N"]
    if not positions:
        raise ValueError("The native sequence has no defined nucleotides.")
    return sum(designed[i] == native[i] for i in positions) / len(positions)
