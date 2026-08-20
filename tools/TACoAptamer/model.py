"""TACoModel：靶标条件化的 RNA 适配体共设计模型（M1-B）

架构（基准结论驱动）：
- RNA 图：GVPEncoder（9 原子含糖环褶皱 + SS 配对边特征）← RhoDesign GVP 复用 + TACo 扩展
- 靶标上下文：配体原子 token（元素 one-hot + 距离 RBF）拼接进 Transformer 序列
- 编码器：fairseq 式 TransformerEncoderLayer ×N（RNA + 配体混合序列）
- 解码器：TransformerDecoder（自回归，只预测 RNA 位置；配体上下文经 cross-attention 注入）
- 损失：交叉熵 + 界面残基加权（结合代理目标 1 的几何版）

坐标约定：rna_coords (B, L, 9, 3)，原子序 [C4', C1', C2', C3', N1, C2, C5', O5', P]
（C1' 在索引 1 = GVP k-NN 中心；前 3 原子参与旋转帧/二面角/侧链特征）
"""
import math
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
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


class TACoModel(nn.Module):
    def __init__(self, args, alphabet, use_official_encoder=False):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.use_official_encoder = use_official_encoder
        embed_dim = args.encoder_embed_dim

        if use_official_encoder:   # 对照模式：完全复用 RhoDesign 官方 encoder
            self._official_enc = self._build_official_encoder(args, alphabet, embed_dim)

        # --- RNA 嵌入组件（RhoDesign 同款） ---
        self.embed_tokens = nn.Embedding(len(alphabet), embed_dim, padding_idx=alphabet.padding_idx)
        # 修复：官方 build_embedding 用 normal(0, embed_dim**-0.5)——nn.Embedding 默认 normal(0,1)
        # 初始化 std 大 22 倍 → decoder 输入嵌入过大 → 训练停滞（诊断 37 号对照定位）
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[alphabet.padding_idx], 0)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, alphabet.padding_idx)
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        # --- GVP（9 原子 + SS 配对边） ---
        gvp_args = argparse.Namespace()
        attrs = {}
        for cls in type(args).__mro__:   # 类属性（type() 创建） + 实例属性都要
            attrs.update(vars(cls))
        attrs.update(vars(args))
        for k, v in attrs.items():
            if k.startswith("gvp_"):
                setattr(gvp_args, k[4:], v)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + 3 * gvp_args.node_hidden_dim_vector
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        # --- 配体上下文嵌入（元素 one-hot + 距离 RBF） ---
        self.embed_ligand = nn.Sequential(
            nn.Linear(5 + 16, embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- 编码器层（处理 RNA + 配体混合序列） ---
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.encoder_layers)
        ])
        self.encoder_layer_norm = nn.LayerNorm(embed_dim)

        # --- 解码器（只预测 RNA） ---
        decoder_embed_tokens = nn.Embedding(
            len(alphabet), args.decoder_embed_dim, padding_idx=alphabet.padding_idx)
        nn.init.normal_(decoder_embed_tokens.weight, mean=0, std=args.decoder_embed_dim ** -0.5)
        nn.init.constant_(decoder_embed_tokens.weight[alphabet.padding_idx], 0)
        self.decoder = TransformerDecoder(args, alphabet, decoder_embed_tokens)

    def _official_encoder_forward(self, rna_coords, lig_feats, lig_mask,
                                 padding_mask, confidence):
        """官方 GVPTransformerEncoder：coords=前 3 原子，adjunct=全原子"""
        from RhoDesign_official.src.gvp_transformer_encoder_without2d import (
            GVPTransformerEncoder)
        c = rna_coords[:, :, :3]
        encoder_out = self._official_enc(c, rna_coords, padding_mask, confidence)
        # 配体 token 拼接（保持三图功能）
        B = rna_coords.size(0)
        n_lig = lig_feats.size(1)
        if n_lig > 0:
            lig_embed = self.embed_ligand(lig_feats) * lig_mask.unsqueeze(-1)
            x = encoder_out["encoder_out"][0]   # (L, B, D)
            x = torch.cat([x, lig_embed.transpose(0, 1)], dim=0)   # (L+n_lig, B, D)
            enc_pm = torch.cat([encoder_out["encoder_padding_mask"][0],
                                ~lig_mask], dim=1)
            encoder_out = {
                "encoder_out": [x],
                "encoder_padding_mask": [enc_pm],
                "encoder_embedding": [],
                "encoder_states": [],
            }
        return encoder_out

    def _build_official_encoder(self, args, alphabet, embed_dim):
        from RhoDesign_official.src.gvp_transformer_encoder_without2d import (
            GVPTransformerEncoder)
        enc_embed_tokens = nn.Embedding(len(alphabet), embed_dim,
                                        padding_idx=alphabet.padding_idx)
        nn.init.normal_(enc_embed_tokens.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(enc_embed_tokens.weight[alphabet.padding_idx], 0)
        return GVPTransformerEncoder(args, alphabet, enc_embed_tokens)

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

        if self.use_official_encoder:
            # 官方 GVPTransformerEncoder 路径（对照实验：完全复用 RhoDesign 官方 encoder）
            return self._official_encoder_forward(rna_coords, lig_feats, lig_mask,
                                                  padding_mask, confidence)

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

        # 位置嵌入：RNA 用 0..L-1，配体续编号 L..L+n_lig-1（共享 Sinusoidal）。
        # 注：fairseq SinusoidalPositionalEmbedding.make_positions 会把值==padding_idx(1)
        # 的位置当 padding(该位置编码为 0,其余 +2 偏移)——训练与推理同路,模型已自适应,
        # 不可修改(改则全部已训 ckpt 失配);仅作已知行为记录(2026-08-19 审查)。
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
        """训练 forward：teacher forcing。
        prev_output_tokens: (B, T) RNA tokens（目标 = 右移一位）
        返回 logits——与 RhoDesign 一致：decoder 输出 (B, V, T)
        （界面加权 interface_mask 参数已随损失加权策略删除,2026-08-19）"""
        encoder_out = self.forward_embedding(
            rna_coords, ss_pair, lig_feats, lig_mask, padding_mask, confidence)
        logits, _ = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return logits

    def sample(self, rna_coords, ss_pair, lig_feats, lig_mask,
               confidence, device, temperature=0.1, n_samples=1):
        """自回归采样（配体上下文固定）。
        提速（2026-08-17）：n_samples>1 时把编码器输出沿 batch 维展开，全部样本并行
        同步解码（原实现逐样本串行，GPU 利用率极低；采样分布完全等价，仅 RNG 流不同，
        池统计不受影响）"""
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

    def set_interface_weight(self, w):
        self.interface_weight = w
