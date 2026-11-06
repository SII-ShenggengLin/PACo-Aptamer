#!/usr/bin/env python3
"""第 38 步：PACoAptamer B 层 test 评估（论文主表口径：池内 best）

- 数据：blayer test 494 链（与 RhoDesign/gRNAde B 层同一评估集）
- 采样：3 温度（0.1/0.5/1.0）× 100 样本 = 300 池内 best（与对比方法同口径）
- 输出：survey/metrics_PACo.json（summary + per_chain）

用法：CUDA_VISIBLE_DEVICES=1 python 38_eval_paco_btest.py --ckpt <路径> --gpu 1
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from PACoAptamer.model import PACoModel
from PACoAptamer.alphabet import Alphabet
from PACoAptamer.data_utils import load_s1

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
TEMPS = [0.1, 0.5, 1.0]
N_SAMPLES = 100
MAX_LEN = 512


def load_model(ckpt_path, device):
    args_cls = type('A', (), {'local_rank': -1, 'device_id': [0],
        'encoder_embed_dim': 512, 'decoder_embed_dim': 512, 'dropout': 0.1,
        'gvp_top_k_neighbors': 15, 'gvp_node_hidden_dim_vector': 256,
        'gvp_node_hidden_dim_scalar': 512, 'gvp_edge_hidden_dim_scalar': 32,
        'gvp_edge_hidden_dim_vector': 1, 'gvp_num_encoder_layers': 3,
        'gvp_dropout': 0.1, 'encoder_layers': 3, 'encoder_attention_heads': 4,
        'attention_dropout': 0.1, 'encoder_ffn_embed_dim': 512,
        'decoder_layers': 3, 'decoder_attention_heads': 4,
        'decoder_ffn_embed_dim': 512})()
    alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])
    model = PACoModel(args_cls, alphabet).cuda(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model, alphabet


@torch.no_grad()
def sample_chain(model, alphabet, d, device, temperature, n_samples, no_ss=False):
    """单链多样本采样（temperature 固定）"""
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = None if no_ss else torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    lig = torch.zeros(1, 0, 21, device=device)
    ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
    seqs = []
    preds = model.sample(coords, ss, lig, ligm, conf, device, temperature=temperature,
                         n_samples=n_samples)   # 批量并行解码（2026-08-17 提速）
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    for row in preds:
        seqs.append(''.join(alphabet.get_tok(t) for t in row.tolist()))
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--quick', type=int, default=0)   # 快速模式：10 样本/温度
    parser.add_argument('--out', type=str, default="metrics_PACo.json")  # 输出名（消融防覆盖，已注释标记）
    parser.add_argument('--no_ss', action="store_true")   # 消融：no_ss 模型评估不喂 SS（已注释标记）
    parser.add_argument('--splits_v2', action="store_true")  # 用 splits_v2.json test 477 链（新划分 v2-B）
    args = parser.parse_args()
    device = 0

    if args.splits_v2:
        sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
        test_stems = list(sp2["test"])
        data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    else:
        blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
        test_stems = [k for k, v in blayer.items() if v == "test"]
        data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_test.pt"))
    test_stems = [s for s in test_stems if s in data]
    print(f"test 链 {len(test_stems)}", flush=True)

    model, alphabet = load_model(args.ckpt, device)
    n_s = args.quick or N_SAMPLES

    results = {}
    t0 = time.time()
    for i, stem in enumerate(test_stems):
        d = data[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        native = d['seq']
        pool = []
        for t in TEMPS:
            pool.extend(sample_chain(model, alphabet, d, device, t, n_s, args.no_ss))
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        results[stem] = {
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": np.mean(recs[:n_s]),
            "n_pool": len(pool),
        }
        if (i + 1) % 50 == 0:
            print(f"进度 {i+1}/{len(test_stems)} | {time.time()-t0:.0f}s", flush=True)

    recs_best = [r["recovery_best"] for r in results.values()]
    recs_t01 = [r["recovery_t01_mean"] for r in results.values()]
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(test_stems)}",
        "recovery_best_mean": float(np.mean(recs_best)),
        "recovery_best_median": float(np.median(recs_best)),
        "recovery_t01_mean_mean": float(np.mean(recs_t01)),
        "temps": TEMPS,
        "n_samples_per_temp": n_s,
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", args.out), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== PACoAptamer B 层 test（{os.path.basename(args.ckpt)}） =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
