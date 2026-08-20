#!/usr/bin/env python3
"""第 27 步：RhoDesign B 层重训（我们的严格划分）

- 模型：RhoDesign_without2d.RhoDesignModel（GVP + Transformer，no_ss 版）
- 数据：B 层 train（3,003 链）/ val（419 链），chain_pdb/ + blayer_splits.json
- 训练：teacher forcing 交叉熵（官方口径），batch_size=1，30 epochs
- 验证：val 集 sample(t=0.1) 恢复率 → best_val checkpoint 保存
- 输出：models/TACoAptamer_37ctl/best_val.pth + current.pth
"""
import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/RhoDesign_official/src")
from TACoAptamer.model import TACoModel as RhoDesignModel
from TACoAptamer.alphabet import Alphabet
from RhoDesign_official.src.util import load_structure, extract_coords_from_structure, CoordBatchConverter

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA_DIR, "chain_pdb")
CKPT_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/TACoAptamer_37ctl"
os.makedirs(CKPT_DIR, exist_ok=True)

class args_class:
    def __init__(self):
        self.local_rank = -1
        self.device_id = [0]
        self.epochs = 5; self.lr = 1e-4; self.batch_size = 1
        self.encoder_embed_dim = 512; self.decoder_embed_dim = 512; self.dropout = 0.1
        # gvp_ 前缀字段（gvp_transformer_encoder 用前缀提取 gvp_args）
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256; self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32; self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3; self.gvp_dropout = 0.1
        self.encoder_layers = 3; self.encoder_attention_heads = 4
        self.attention_dropout = 0.1; self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3; self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512

def load_chain(stem):
    """加载单链 → (coords, seq)；序列归一化（I→G，其他非 ACGU→X）"""
    pdb_path = os.path.join(CHAIN_DIR, f"{stem}.pdb")
    structure = load_structure(pdb_path)
    coords, seq = extract_coords_from_structure(structure)
    seq = "".join({"I": "G"}.get(c, c if c in "ACGU" else "X") for c in seq)
    return coords, seq

def get_data(stems):
    """特征化全部链（可缓存）"""
    cache = os.path.join(DATA_DIR, "grnade_btrain", "rhodesign_feats.npz")
    data = {}
    for stem in stems:
        coords, seq = load_chain(stem)
        data[stem] = (coords.astype(np.float32), seq)
    return data

def train_epoch(model, data, stems, batch_converter, alphabet, optimizer, device, shuffle=True):
    model.train()
    total_loss, total_tok = 0.0, 0
    order = list(stems)
    if shuffle:
        random.shuffle(order)
    for stem in order:
        coords, seq = data[stem]
        L = len(seq)
        if L > 512:
            continue
        batch_coords, confidence, strs, tokens, padding_mask, _ = batch_converter(
            [(coords, None, seq, None)])  # seq 必传：tokens 编码真实序列
        tokens = tokens.to(device)
        padding_mask = padding_mask.bool().to(device)
        batch_coords = batch_coords.to(device)
        confidence = confidence.to(device)
        # teacher forcing：decoder 输入 = tokens[:-1]，目标 = tokens[1:]
        # 注意：decoder 输出 logits 形状 (B, V, T)（转置），需 permute
        logits = model(batch_coords, None,
                       torch.zeros(1, 0, 21, device=device),
                       torch.zeros(1, 0, dtype=torch.bool, device=device),
                       padding_mask, confidence,
                       prev_output_tokens=tokens[:, :-1])
        logits = logits.permute(0, 2, 1)  # (B, T, V)
        target = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1), ignore_index=alphabet.padding_idx)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * (target != alphabet.padding_idx).sum().item()
        total_tok += (target != alphabet.padding_idx).sum().item()
    return total_loss / max(1, total_tok)

def eval_recovery(model, data, stems, batch_converter, alphabet, device, temperature=0.1):
    """val 采样恢复率。两个 2026-08-19 修复：
    ① pred 是 token 索引张量,必须 alphabet.get_idx(b) == a 比较(曾与字符比较恒 False
       → VAL recovery 恒 0,best_val 选择失效);
    ② 评估输入必须与训练同路:经 CoordBatchConverter(首尾各 pad 一个哑残基;
       曾手工构造 (1,L,·,3) 绕过 converter → 位置编码/键数偏移)。
    converter 坐标长 L+2(首尾哑残基),sample 解码 L+2 步 → 取前 L 位对齐真实残基。"""
    model.eval()
    recs = []
    with torch.no_grad():
        for stem in stems:
            coords, seq = data[stem]
            L = len(seq)
            if L > 512:
                continue
            batch_coords, confidence, _, _, _, _ = batch_converter(
                [(coords, None, seq, None)])
            pred = model.sample(batch_coords.to(device), None,
                                torch.zeros(1, 0, 21, device=device),
                                torch.zeros(1, 0, dtype=torch.bool, device=device),
                                confidence.to(device), device, temperature=temperature)
            pr = pred.tolist()
            pr = pr[:L] if isinstance(pr[0], int) else pr[0][:L]
            rec = sum(1 for a, b in zip(pr, seq)
                      if alphabet.get_idx(b) == a) / max(1, L)
            recs.append(rec)
    return np.mean(recs) if recs else 0.0

def main(gpu):
    _ = gpu  # CUDA_VISIBLE_DEVICES 已限制，设备固定 cuda:0
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = 0  # CUDA_VISIBLE_DEVICES 限制后设备号从 0 数

    blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
    train_stems = [k for k, v in blayer.items() if v == "train"]
    val_stems = [k for k, v in blayer.items() if v == "val"]
    print(f"train {len(train_stems)} / val {len(val_stems)}", flush=True)

    print("特征化训练数据...", flush=True)
    train_data = get_data(train_stems)
    val_data = get_data(val_stems)
    print(f"特征化完成: train {len(train_data)} / val {len(val_data)}", flush=True)

    args = args_class()
    dictionary = Alphabet(["A", "G", "C", "U", "X"])
    import functools
    model = functools.partial(RhoDesignModel, use_official_encoder=True)(args, dictionary).cuda(0)
    batch_converter = CoordBatchConverter(dictionary)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_rec = -1
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_data, train_stems, batch_converter,
                           dictionary, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            rec = eval_recovery(model, val_data, val_stems, batch_converter, dictionary, device)
            print(f"Epoch {epoch+1}: loss {loss:.4f} | VAL recovery {rec:.4f}", flush=True)
            if rec > best_rec:
                best_rec = rec
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_val.pth"))
                print(f"  → best 保存 ({best_rec:.4f})", flush=True)
        else:
            print(f"Epoch {epoch+1}: loss {loss:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "current.pth"))

    print(f"训练完成: best VAL recovery {best_rec:.4f}", flush=True)
    print(f"checkpoint: {CKPT_DIR}/best_val.pth", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    main(args.gpu)
