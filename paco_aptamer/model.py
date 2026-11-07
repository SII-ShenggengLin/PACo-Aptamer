"""Ligand-conditioned RNA inverse-folding model.

RNA coordinates have shape (B, L, 9, 3) and use the atom order defined by
``paco_aptamer.data.PACO9``.
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvp_src.gvp_encoder import GVPEncoder
from .gvp_src.features import GVPInputFeaturizer, DihedralFeatures
from .transformer_src.transformer_decoder import TransformerDecoder
from .transformer_src.transformer_layer import (
    TransformerEncoderLayer, SinusoidalPositionalEmbedding,
)


def rbf(D, v_min, v_max, n_bins=16):
    """Distance RBF embedding"""
    D = D.clamp(v_min, v_max)
    d = (D - v_min) / (v_max - v_min)
    mu = torch.linspace(0, 1, n_bins, device=D.device).view(1, 1, -1)
    sigma = 1.0 / n_bins
    return torch.exp(-((d.unsqueeze(-1) - mu) ** 2) / (2 * sigma ** 2))


def nan_to_num(t, nan=0.0, posinf=1e5, neginf=-1e5):
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)


class PACoModel(nn.Module):
    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        embed_dim = args.encoder_embed_dim

        self.embed_tokens = nn.Embedding(
            len(alphabet), embed_dim, padding_idx=alphabet.padding_idx
        )
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[alphabet.padding_idx], 0)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, alphabet.padding_idx)
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        gvp_args = argparse.Namespace()
        attrs = {}
        for cls in type(args).__mro__:
            attrs.update(vars(cls))
        attrs.update(vars(args))
        for key, value in attrs.items():
            if key.startswith("gvp_"):
                setattr(gvp_args, key[4:], value)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + 3 * gvp_args.node_hidden_dim_vector
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.embed_ligand = nn.Sequential(
            nn.Linear(5 + 16, embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.encoder_layers)
        ])
        self.encoder_layer_norm = nn.LayerNorm(embed_dim)

        decoder_embed_tokens = nn.Embedding(
            len(alphabet), args.decoder_embed_dim, padding_idx=alphabet.padding_idx
        )
        nn.init.normal_(
            decoder_embed_tokens.weight,
            mean=0,
            std=args.decoder_embed_dim ** -0.5,
        )
        nn.init.constant_(decoder_embed_tokens.weight[alphabet.padding_idx], 0)
        self.decoder = TransformerDecoder(args, alphabet, decoder_embed_tokens)

    def forward_embedding(self, rna_coords, ss_pair, lig_feats, lig_mask,
                          padding_mask, confidence):
        """RNA + 配体混合序列嵌入。
        rna_coords: (B, L, 9, 3)；ss_pair: (B, L, L) bool
        lig_feats: (B, n_lig, 21)；lig_mask: (B, n_lig) bool（True=有效）
        padding_mask: (B, L) True=padding（RNA 侧）
        """
        B, L = rna_coords.size(0), rna_coords.size(1)
        n_lig = lig_feats.size(1)
        device = rna_coords.device

        coord_mask = torch.all(torch.all(torch.isfinite(rna_coords), dim=-1), dim=-1)
        rna_coords = nan_to_num(rna_coords)
        mask_tokens = (
            padding_mask * self.padding_idx +
            ~padding_mask * self.alphabet.get_idx("<mask>")
        )

        # RNA 嵌入
        rna_embed = self.embed_tokens(mask_tokens) * self.embed_scale
        rna_embed = rna_embed + self.embed_dihedrals(rna_coords)

        if ss_pair is None:   # 无 SS 时传全 0（维度一致，特征无信息）
            ss_pair = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(
            rna_coords, coord_mask, padding_mask, confidence, ss_pair)
        from .util import rotate, get_rotation_frames
        R = get_rotation_frames(rna_coords)
        gvp_out = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        rna_embed = rna_embed + self.embed_gvp_output(gvp_out)

        rna_embed = rna_embed + self.embed_confidence(rbf(confidence, 0., 1.))

        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            rna_coords, coord_mask, with_coord_mask=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        rna_embed = rna_embed + self.embed_gvp_input_features(features)

        # 配体嵌入（上下文 token）
        if n_lig > 0:
            lig_embed = self.embed_ligand(lig_feats)   # (B, n_lig, D)
            lig_embed = lig_embed * lig_mask.unsqueeze(-1)
        else:
            lig_embed = torch.zeros(B, 0, self.args.encoder_embed_dim, device=device)

        x = torch.cat([rna_embed, lig_embed], dim=1)   # (B, L+n_lig, D)

        # Preserve the positional-indexing convention used by the released checkpoints.
        all_positions = torch.arange(L + n_lig, device=device).view(1, -1).expand(B, -1)
        x = x + self.embed_positions(all_positions)

        # 编码器 padding mask：RNA padding + 配体全部有效（参与 attention）
        enc_padding_mask = torch.cat([
            padding_mask,
            ~lig_mask if n_lig > 0 else torch.zeros(B, 0, dtype=torch.bool, device=device),
        ], dim=1)

        # 编码器层（fairseq T-first 约定：B×T×C → T×B×C）
        x = x * (1 - enc_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, enc_padding_mask)
        x = self.encoder_layer_norm(x)

        # encoder_out 格式（fairseq 兼容：dict of lists，T-first）
        encoder_out = {
            "encoder_out": [x],                      # (L+n_lig, B, D)
            "encoder_padding_mask": [enc_padding_mask],
            "encoder_embedding": [],
            "encoder_states": [],
        }
        return encoder_out

    def forward(self, rna_coords, ss_pair, lig_feats, lig_mask,
                padding_mask, confidence, prev_output_tokens):
        """Return teacher-forced logits with shape (B, vocabulary, T)."""
        encoder_out = self.forward_embedding(
            rna_coords, ss_pair, lig_feats, lig_mask, padding_mask, confidence)
        logits, _ = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return logits

    def sample(self, rna_coords, ss_pair, lig_feats, lig_mask,
               confidence, device, temperature=0.1, n_samples=1):
        """Sample one or more sequences with batched autoregressive decoding."""
        self.eval()
        with torch.no_grad():
            L = rna_coords.size(0) if rna_coords.dim() == 3 else rna_coords.size(1)
            if rna_coords.dim() == 3:
                rna_coords = rna_coords.unsqueeze(0)
                ss_pair = ss_pair.unsqueeze(0) if ss_pair is not None else None
                lig_feats = lig_feats.unsqueeze(0) if lig_feats is not None else torch.zeros(1, 0, 21, device=device)
                lig_mask = lig_mask.unsqueeze(0) if lig_mask is not None else torch.zeros(1, 0, dtype=torch.bool, device=device)
                confidence = confidence.unsqueeze(0) if confidence is not None else torch.ones(1, L, device=device)
            B, L = rna_coords.size(0), rna_coords.size(1)
            n = int(n_samples)
            assert B == 1 or n == 1, "批量采样仅支持单结构"
            padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            encoder_out = self.forward_embedding(
                rna_coords, ss_pair, lig_feats, lig_mask, padding_mask, confidence)
            if n > 1:
                encoder_out = {
                    "encoder_out": [encoder_out["encoder_out"][0].repeat(1, n, 1)],
                    "encoder_padding_mask": [encoder_out["encoder_padding_mask"][0].repeat(n, 1)],
                    "encoder_embedding": [],
                    "encoder_states": [],
                }
            NB = n if n > 1 else B
            bos = self.alphabet.get_idx("<cath>")
            pred = torch.zeros(NB, L + 1, dtype=torch.long, device=device)
            pred[:, 0] = bos   # BOS 起始（与训练一致）
            state = {}
            for i in range(L):
                prev = pred[:, :i + 1]
                logits, _ = self.decoder(prev, encoder_out=encoder_out,
                                         incremental_state=state)
                logits = logits[:, :, -1]   # fairseq (B, V, T) → 最后位置 (B, V)
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    sample = logits.argmax(-1)
                pred[:, i + 1] = sample
            return pred[:, 1:] if n > 1 else pred[0, 1:]