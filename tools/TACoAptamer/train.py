"""TACoAptamer 训练：S1 纯 RNA 预训练 → S2 复合物微调

- S1：splits_v2 train 5,773 链（taco_s1_v2.pt 特征）+ SS 条件 + 数据增强（噪声+旋转）
- S2：s2_samples_v2 复合物链（配体上下文条件化）+ joint 混训防遗忘 + val_sel 选 ckpt
- 最终主配置 ens2-noiface：S2 统一配方 joint + val_sel + batch4 + 8 epoch（无界面加权，
  该策略已删——P0-0b 决策，analysis/13）
- 输出：models/TACoAptamer/{out}.pth

用法：CUDA_VISIBLE_DEVICES=1 python train.py --stage s2 --init <s1.pth> \
  --splits_v2 --s2_data s2_samples_v2.pt --joint --val_sel --epochs 8 --seed 43 \
  --out s2_v2_seed43_noiface
"""
import os, sys, json, random, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from TACoAptamer.model import TACoModel
from TACoAptamer.alphabet import Alphabet
from TACoAptamer.data_utils import load_s1, load_s2, augment

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CKPT_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/TACoAptamer"
os.makedirs(CKPT_DIR, exist_ok=True)

ATOM_IDX_C1P = 1   # TACo 9 原子中 C1' 位置


class args_class:
    def __init__(self):
        self.local_rank = -1
        self.device_id = [0]
        self.lr = 1e-4
        self.encoder_embed_dim = 512
        self.decoder_embed_dim = 512
        self.dropout = 0.1
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256
        self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32
        self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3
        self.gvp_dropout = 0.1
        self.encoder_layers = 3
        self.encoder_attention_heads = 4
        self.attention_dropout = 0.1
        self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3
        self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512


def to_tensor(d, device):
    coords = torch.tensor(d["coords9"], dtype=torch.float32, device=device).unsqueeze(0)
    ss = torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
    seq = d["seq"]
    return coords, ss, seq


def train_epoch_s1_batched(model, data, stems, alphabet, optimizer, device, max_len=512,
                           no_ss=False, no_aug=False, batch_chains=8, ss_dropout=0.0):
    """批量版 S1 训练（2026-08-17）：按长度分桶 + padding 对齐，batch_chains 链/步。
    动机：逐链步进 46ms/链，其中 GPU 计算 <10ms，其余是 kernel 启动/优化器/CPU 转换开销；
    批量后摊薄。动力学说明：token 均值 CE 与逐链版同口径，但步数/epoch ↓batch_chains 倍，
    需相应增加 epoch 或 LR（--batch_chains 显式开启，默认 1 = 旧行为）。"""
    model.train()
    total_loss, total_tok = 0.0, 0
    pad_tok = alphabet.padding_idx
    bos = alphabet.get_idx("<cath>")
    # 长度分桶（减少 padding 浪费），桶间 shuffle
    order = sorted((s for s in stems if len(data[s]["seq"]) <= max_len),
                   key=lambda s: len(data[s]["seq"]))
    batches = [order[i:i + batch_chains] for i in range(0, len(order), batch_chains)]
    random.shuffle(batches)
    for batch_stems in batches:
        B = len(batch_stems)
        L_max = max(len(data[s]["seq"]) for s in batch_stems)
        coords = torch.zeros(B, L_max, 9, 3, dtype=torch.float32, device=device)
        ss = torch.zeros(B, L_max, L_max, dtype=torch.bool, device=device)
        tokens = torch.full((B, L_max + 1), pad_tok, dtype=torch.long, device=device)
        padding_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        confidence = torch.ones(B, L_max, device=device)
        tokens[:, 0] = bos
        for j, s in enumerate(batch_stems):
            d = data[s]
            L = len(d["seq"])
            c = augment(d["coords9"]) if not no_aug else d["coords9"]
            coords[j, :L] = torch.tensor(c, dtype=torch.float32, device=device)
            coords[j, L:] = torch.tensor(c[0:1], dtype=torch.float32, device=device)  # 有限值填充（防 dihedral NaN），被 mask
            if not no_ss:
                # SS dropout（Phase C，2026-08-17）：以 ss_dropout 概率整链置零 SS，
                # 防模型对 SS 过依赖（短链/少配对链鲁棒性，直击 30-60nt 短板）
                if ss_dropout > 0 and random.random() < ss_dropout:
                    pass   # ss 块保持全零
                else:
                    ss[j, :L, :L] = torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device)
            tokens[j, 1:L + 1] = torch.tensor([alphabet.get_idx(c2) for c2 in d["seq"]],
                                              dtype=torch.long, device=device)
            padding_mask[j, :L] = False
        lig_feats = torch.zeros(B, 0, 21, device=device)
        lig_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
        logits = model(coords, None if no_ss else ss, lig_feats, lig_mask,
                       padding_mask, confidence, prev_output_tokens=tokens[:, :-1])
        logits = logits.permute(0, 2, 1)
        target = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1), ignore_index=pad_tok)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_tok = (target != pad_tok).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
    return total_loss / max(1, total_tok)


def train_epoch_s2_batched(model, data, stems, alphabet, optimizer, device, max_len=512,
                           no_ss=False, no_aug=False, no_ligand=False,
                           s1_data=None, s1_stems=None, batch_chains=8, ss_dropout=0.0):
    """S2 微调 epoch（批量版，唯一训练路径；逐链版已删）。
    长度分桶 + padding；配体按批内最大原子数对齐（lig_mask 标记有效）；
    joint 模式混入等量 S1 链（无配体）。
    注：界面加权损失 ×3 策略已于 2026-08-18 P0-0b 决策删除（界面口径无显著增益，
    且旧实现对齐有 off-by-one）；本函数为普通 token 均值 CE。"""
    model.train()
    total_loss, total_tok = 0.0, 0
    pad_tok = alphabet.padding_idx
    bos = alphabet.get_idx("<cath>")
    if s1_data is not None:
        s1_cur = list(s1_stems)
        random.shuffle(s1_cur)
        s1_cur = s1_cur[:len(stems)]
    else:
        s1_cur = []
    s1_set = set(s1_cur)
    all_stems = [s for s in (list(stems) + s1_cur)
                 if len((s1_data if s in s1_set else data)[s]["seq"]) <= max_len]
    order = sorted(all_stems, key=lambda s: len((s1_data if s in s1_set else data)[s]["seq"]))
    batches = [order[i:i + batch_chains] for i in range(0, len(order), batch_chains)]
    random.shuffle(batches)
    for batch_stems in batches:
        B = len(batch_stems)
        get = lambda s: (s1_data if s in s1_set else data)[s]
        L_max = max(len(get(s)["seq"]) for s in batch_stems)
        has_lig = (not no_ligand) and any(s not in s1_set and get(s)["lig_feats"].shape[0] > 0
                                          for s in batch_stems)
        n_lig_max = max([get(s)["lig_feats"].shape[0] if (not no_ligand and s not in s1_set) else 0
                         for s in batch_stems]) if has_lig else 0
        coords = torch.zeros(B, L_max, 9, 3, dtype=torch.float32, device=device)
        ss = torch.zeros(B, L_max, L_max, dtype=torch.bool, device=device)
        tokens = torch.full((B, L_max + 1), pad_tok, dtype=torch.long, device=device)
        padding_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        confidence = torch.ones(B, L_max, device=device)
        lig_feats = torch.zeros(B, n_lig_max, 21, device=device)
        lig_mask = torch.zeros(B, n_lig_max, dtype=torch.bool, device=device)
        tokens[:, 0] = bos
        for j, s in enumerate(batch_stems):
            is_s1 = s in s1_set
            d = get(s)
            L = len(d["seq"])
            c = augment(d["coords9"]) if not no_aug else d["coords9"]
            coords[j, :L] = torch.tensor(c, dtype=torch.float32, device=device)
            coords[j, L:] = torch.tensor(c[0:1], dtype=torch.float32, device=device)
            if not no_ss and not (ss_dropout > 0 and random.random() < ss_dropout):
                ss[j, :L, :L] = torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device)
            tokens[j, 1:L + 1] = torch.tensor([alphabet.get_idx(c2) for c2 in d["seq"]],
                                              dtype=torch.long, device=device)
            padding_mask[j, :L] = False
            if not no_ligand and not is_s1 and d["lig_feats"].shape[0] > 0:
                nl = d["lig_feats"].shape[0]
                lig_feats[j, :nl] = torch.tensor(d["lig_feats"], dtype=torch.float32, device=device)
                lig_mask[j, :nl] = True
        logits = model(coords, None if no_ss else ss, lig_feats, lig_mask,
                       padding_mask, confidence, prev_output_tokens=tokens[:, :-1])
        logits = logits.permute(0, 2, 1)
        target = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1), ignore_index=pad_tok)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_tok = (target != pad_tok).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
    return total_loss / max(1, total_tok)


@torch.no_grad()
def eval_recovery(model, data, stems, alphabet, device, temperature=0.1, max_len=512,
                  no_ss=False):
    model.eval()
    recs = []
    for stem in stems:
        d = data[stem]
        L = len(d["seq"])
        if L > max_len:
            continue
        coords = torch.tensor(d["coords9"], dtype=torch.float32, device=device).unsqueeze(0)
        ss = None if no_ss else torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
        confidence = torch.ones(1, L, device=device)
        lig_feats = torch.zeros(1, 0, 21, device=device)
        lig_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pred = model.sample(coords, ss, lig_feats, lig_mask, confidence, device,
                            temperature=temperature)
        rec = sum(1 for a, b in zip(pred.tolist(), d["seq"])
                  if alphabet.get_idx(b) == a) / max(1, len(d["seq"]))
        recs.append(rec)
    return np.mean(recs) if recs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s1", "s2"], required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init", type=str, default=None)   # S2 用 S1 checkpoint
    parser.add_argument("--no_ss", action="store_true")     # 消融：无 SS 配对条件
    parser.add_argument("--no_aug", action="store_true")    # 消融：无数据增强
    parser.add_argument("--no_ligand", action="store_true") # S2 消融：无配体上下文
    parser.add_argument("--val_eval", action="store_true")  # S2 完成后自动 val 评估
    parser.add_argument("--val_sel", action="store_true")   # S2 按 val recovery 选 ckpt（防 train-loss 过拟合选择，2026-08-16）
    parser.add_argument("--out", type=str, default=None)    # checkpoint 名（消融防覆盖）
    parser.add_argument("--lr", type=float, default=None)    # 学习率覆盖（S2 反遗忘调参）
    parser.add_argument("--joint", action="store_true")      # S2 混合 S1 训练防遗忘（1:1 抽样）
    parser.add_argument("--splits_v2", action="store_true")  # 用 splits_v2.json（新划分 v2-B）
    parser.add_argument("--s2_data", type=str, default="s2_samples.pt")   # S2 源数据（干净版用 s2_samples_clean.pt）
    parser.add_argument("--seed", type=int, default=42)      # 随机种子（集成实验用）
    parser.add_argument("--batch_chains", type=int, default=4)  # 批量链数/步（长度分桶批量训练）
    parser.add_argument("--ss_dropout", type=float, default=0.0)  # SS 整链丢弃率（防 SS 过依赖）
    args = parser.parse_args()

    random.seed(args.seed)   # 种子可配（2026-08-17，集成实验用不同种子）
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 0   # CUDA_VISIBLE_DEVICES 限制后从 0 数

    model_args = args_class()
    if args.lr:
        model_args.lr = args.lr
    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = TACoModel(model_args, alphabet).cuda(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"初始化自 {args.init}", flush=True)
    optimizer = Adam(model.parameters(), lr=model_args.lr)

    if args.splits_v2:   # 新划分 v2-B（splits_v2.json 分组列表；train 5,773 / val 96）
        sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
        train_stems = list(sp2["train"])
        val_stems = list(sp2["val"])
    else:
        blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
        train_stems = [k for k, v in blayer.items() if v == "train"]
        val_stems = [k for k, v in blayer.items() if v == "val"]

    out_name = args.out or ("s1_best" if args.stage == "s1" else "s2_best")
    out_path = os.path.join(CKPT_DIR, out_name + ".pth")

    if args.stage == "s1":
        print("加载 S1 数据...", flush=True)
        s1_cache = "taco_s1_v2.pt" if args.splits_v2 else "taco_s1.pt"
        val_cache = "taco_val.pt" if not args.splits_v2 else "taco_s1_v2.pt"
        data = load_s1(set(train_stems), cache=os.path.join(DATA_DIR, s1_cache))
        val_data = load_s1(set(val_stems), cache=os.path.join(DATA_DIR, val_cache))
        train_stems = [s for s in train_stems if s in data]   # processed.pt 可能缺链（特征化失败）
        val_stems = [s for s in val_stems if s in val_data]
        print(f"S1 实际可用: train {len(train_stems)} / val {len(val_stems)}", flush=True)
        epochs = args.epochs or 30
        best_rec = -1
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_epoch_s1_batched(model, data, train_stems, alphabet, optimizer, device,
                                          no_ss=args.no_ss, no_aug=args.no_aug,
                                          batch_chains=args.batch_chains,
                                          ss_dropout=args.ss_dropout)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                rec = eval_recovery(model, val_data, val_stems, alphabet, device,
                                    no_ss=args.no_ss)
                print(f"S1 Epoch {epoch+1}: loss {loss:.4f} | VAL {rec:.4f} | {time.time()-t0:.0f}s", flush=True)
                if rec > best_rec:
                    best_rec = rec
                    torch.save(model.state_dict(), out_path)
                    print(f"  → s1_best 保存 ({best_rec:.4f})", flush=True)
            else:
                print(f"S1 Epoch {epoch+1}: loss {loss:.4f} | {time.time()-t0:.0f}s", flush=True)
        print(f"S1 完成: best VAL {best_rec:.4f}", flush=True)
    else:
        print("加载 S2 数据...", flush=True)
        # cache 名按源文件名映射（v2 剔除版用独立缓存，避免误读旧 clean 版）
        if args.s2_data == "s2_samples.pt":
            s2_cache = os.path.join(DATA_DIR, "taco_s2.pt")
        elif args.s2_data == "s2_samples_clean.pt":
            s2_cache = os.path.join(DATA_DIR, "taco_s2_clean.pt")
        else:   # 新划分 v2（s2_samples_v2.pt）
            s2_cache = os.path.join(DATA_DIR, "taco_s2_v2.pt")
        data = load_s2(cache=s2_cache, src=args.s2_data)
        stems = list(data.keys())
        s1_data, s1_stems = None, None
        if args.joint:   # 混合 S1 训练（反遗忘）
            print("S2-joint: 混入 S1 train 数据...", flush=True)
            s1_data = load_s1(set(train_stems), cache=os.path.join(
                DATA_DIR, "taco_s1_v2.pt" if args.splits_v2 else "taco_s1.pt"))
            s1_stems = [s for s in train_stems if s in s1_data]
            print(f"  S1 可用 {len(s1_stems)} 链", flush=True)
        epochs = args.epochs or 15
        best_loss = 1e9
        # 修复（2026-08-16）：S2 原按 train loss 选 ckpt —— 859 链小集 15 epoch 必然选到
        # 最过拟合的 epoch（train loss 0.13 = 记忆化，采样多样性被摧毁，短链 mode-lock）。
        # 改为 --val_sel 时按 val recovery（无配体口径）选最优 epoch；未开时保留旧行为。
        val_data_s2, val_ok_s2 = None, None
        if args.val_sel:
            val_data_s2 = load_s1(set(val_stems), cache=os.path.join(
                DATA_DIR, "taco_val.pt" if not args.splits_v2 else "taco_s1_v2.pt"))
            val_ok_s2 = [s for s in val_stems if s in val_data_s2]
            # 修复（2026-08-17）：splits_v2 val 有 37/96 链与 S2 复合物同 PDB（审计曾只查
            # test/indep/zs 漏了 val）→ S2 微调会记忆这些链，val 选择被污染（虚高 ~0.71）。
            # 选 ckpt 用干净 val（剔除与 S2 同 PDB 的链，剩 59 链）
            s2_pdbs = {k.split("_")[0] for k in data.keys()}
            n0 = len(val_ok_s2)
            val_ok_s2 = [s for s in val_ok_s2 if s.split("_")[0] not in s2_pdbs]
            print(f"val_sel 干净 val: {len(val_ok_s2)}/{n0} 链（剔除 S2 同 PDB 链）", flush=True)
        best_rec = -1.0
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_epoch_s2_batched(model, data, stems, alphabet, optimizer, device,
                                          no_ss=args.no_ss, no_aug=args.no_aug,
                                          no_ligand=args.no_ligand,
                                          s1_data=s1_data, s1_stems=s1_stems,
                                          batch_chains=args.batch_chains,
                                          ss_dropout=args.ss_dropout)
            if args.val_sel:
                rec = eval_recovery(model, val_data_s2, val_ok_s2, alphabet, device,
                                    no_ss=args.no_ss)
                print(f"S2 Epoch {epoch+1}: loss {loss:.4f} | VAL {rec:.4f} | {time.time()-t0:.0f}s", flush=True)
                if rec > best_rec:
                    best_rec = rec
                    torch.save(model.state_dict(), out_path)
                    print(f"  → s2_best 保存 (VAL {best_rec:.4f})", flush=True)
            else:
                print(f"S2 Epoch {epoch+1}: loss {loss:.4f} | {time.time()-t0:.0f}s", flush=True)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), out_path)
                    print(f"  → s2_best 保存 ({best_loss:.4f})", flush=True)
        print(f"S2 完成: best {'VAL ' + format(best_rec, '.4f') if args.val_sel else 'loss ' + format(best_loss, '.4f')}", flush=True)
        if args.val_eval:   # 消融对比：S2 后 val 恢复率（无配体输入口径）
            val_data = load_s1(set(val_stems), cache=os.path.join(
                DATA_DIR, "taco_val.pt" if not args.splits_v2 else "taco_s1_v2.pt"))
            val_ok = [s for s in val_stems if s in val_data]
            rec = eval_recovery(model, val_data, val_ok, alphabet, device, no_ss=args.no_ss)
            print(f"S2 val recovery: {rec:.4f}", flush=True)


if __name__ == "__main__":
    main()
