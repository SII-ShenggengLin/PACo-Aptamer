#!/usr/bin/env python3
"""对照实验：PACoModel × RhoDesign 特征管线（biotite 7 原子，27 号同款）

目的：隔离"我们的数据管线（processed.pt 9 原子）" vs "模型实现"——若 biotite 版学得好
（VAL 显著 >0.5），则数据管线有问题；否则模型实现有 bug。
- 数据：blayer train/val，biotite 提取 7 原子 [C4', C1', N1, C2, C5', O5', P]
- SS：RNAfold（从 processed.pt 取，链级对应）
- 训练：与 train.py 相同循环，5 epoch 快速验证
用法：CUDA_VISIBLE_DEVICES=2 python 36_ablation_s1_biotite.py
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from PACoAptamer.model import PACoModel
PACoModel = lambda a, b: __import__("PACoAptamer.model", fromlist=["PACoModel"]).PACoModel(a, b, use_official_encoder=True)
from PACoAptamer.alphabet import Alphabet
from PACoAptamer.data_utils import augment

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
RHO_SRC = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/RhoDesign_official/src"

no_ss = False   # 全局开关（--no_ss 时 True）
no_aug = False  # 全局开关（--no_aug 时 True，对照 RhoDesign 无增强）

# biotite 7 原子（RhoDesign 官方顺序）
ATOMS7 = ["C4'", "C1'", "N1", "C2", "C5'", "O5'", "P"]


def load_biotite(stems):
    """biotite 提取 (L, 7, 3) + 序列；SS 从 processed.pt 对应链取"""
    sys.path.insert(0, RHO_SRC)
    from util import load_structure, extract_coords_from_structure
    proc = torch.load(os.path.join(DATA_DIR, "grnade_btrain", "processed.pt"),
                      map_location="cpu", weights_only=False)
    out = {}
    for stem in stems:
        pdb_path = os.path.join(DATA_DIR, "chain_pdb", f"{stem}.pdb")
        if not os.path.exists(pdb_path):
            continue
        structure = load_structure(pdb_path)
        coords, seq = extract_coords_from_structure(structure)
        seq = "".join({"I": "G"}.get(c, c if c in "ACGU" else "X") for c in seq)
        if len(coords) != len(seq) or len(seq) > 512:
            continue
        p = proc.get(stem)
        if p is None:
            continue
        ss = p["sec_struct_list"][0]
        if len(ss) != len(seq):
            continue
        from PACoAptamer.data_utils import ss_to_pairs
        out[stem] = {
            "coords": coords.astype(np.float32),   # (L, 7, 3)
            "ss_pairs": ss_to_pairs(ss),
            "seq": seq,
        }
    return out


class args_class:
    def __init__(self):
        self.local_rank = -1; self.device_id = [0]
        self.encoder_embed_dim = 512; self.decoder_embed_dim = 512; self.dropout = 0.1
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256; self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32; self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3; self.gvp_dropout = 0.1
        self.encoder_layers = 3; self.encoder_attention_heads = 4
        self.attention_dropout = 0.1; self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3; self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512


def train_epoch(model, data, stems, alphabet, optimizer, device):
    model.train()
    total_loss, total_tok = 0.0, 0
    random.shuffle(stems)
    for stem in stems:
        d = data[stem]
        L = len(d["seq"])
        if L > 512:
            continue
        coords = d["coords"] if no_aug else augment(d["coords"])
        coords = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        ss = None if no_ss else torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
        bos = alphabet.get_idx("<cath>")
        tokens = torch.tensor([[bos] + [alphabet.get_idx(c) for c in d["seq"]]], device=device)
        pm = torch.zeros(1, L, dtype=torch.bool, device=device)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        logits = model(coords, ss, lig, ligm, pm, conf, prev_output_tokens=tokens[:, :-1])
        logits = logits.permute(0, 2, 1)
        target = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1), ignore_index=alphabet.padding_idx)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_tok = (target != alphabet.padding_idx).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
    return total_loss / max(1, total_tok)


@torch.no_grad()
def eval_recovery(model, data, stems, alphabet, device, temperature=0.1):
    model.eval()
    recs = []
    for stem in stems:
        d = data[stem]
        L = len(d["seq"])
        if L > 512:
            continue
        coords = torch.tensor(d["coords"], dtype=torch.float32, device=device).unsqueeze(0)
        ss = None if no_ss else torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pred = model.sample(coords, ss, lig, ligm, conf, device, temperature=temperature)
        rec = sum(1 for a, b in zip(pred.tolist(), d["seq"])
                  if alphabet.get_idx(b) == a) / max(1, len(d["seq"]))
        recs.append(rec)
    return np.mean(recs) if recs else 0.0


def main():
    global no_ss, no_aug
    no_ss = "--no_ss" in sys.argv
    no_aug = "--no_aug" in sys.argv
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = 0
    blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
    train_stems = [k for k, v in blayer.items() if v == "train"]
    val_stems = [k for k, v in blayer.items() if v == "val"]
    print("biotite 特征化...", flush=True)
    data = load_biotite(train_stems)
    val_data = load_biotite(val_stems)
    train_stems = [s for s in train_stems if s in data]
    val_stems = [s for s in val_stems if s in val_data]
    print(f"train {len(train_stems)} / val {len(val_stems)}", flush=True)

    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = PACoModel(args_class(), alphabet).cuda(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    for epoch in range(5):
        t0 = time.time()
        loss = train_epoch(model, data, train_stems, alphabet, optimizer, device)
        rec = eval_recovery(model, val_data, val_stems, alphabet, device)
        print(f"Epoch {epoch+1}: loss {loss:.4f} | VAL {rec:.4f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
