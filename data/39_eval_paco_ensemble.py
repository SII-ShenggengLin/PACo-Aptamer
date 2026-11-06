#!/usr/bin/env python3
"""第 39 步：PACoAptamer 集成评估（K 模型逐步 logits 平均采样）

- 与 38 号同口径：splits_v2 test 477 链，3 温度 × 100 样本池内 best / t01 均值
- 区别：自回归解码每步对 K 个 ckpt 的 softmax 概率取平均后 multinomial
  （等价于混合分布采样；池大小不变 = 公平口径，推理代价 ×K）
- 用法：CUDA_VISIBLE_DEVICES=2 python data/39_eval_paco_ensemble.py \
    --ckpts s2_v2_seed43,s2_v2_ssdo15 --gpu 0 --splits_v2 --out metrics_PACo_v2_ens2_full.json
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data")
from PACoAptamer.data_utils import load_s1
import importlib.util
spec = importlib.util.spec_from_file_location(
    "m38", "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data/38_eval_paco_btest.py")
m38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m38)   # 复用 load_model / TEMPS / N_SAMPLES / MAX_LEN / DATA_DIR


@torch.no_grad()
def sample_chain_ens(models, alphabet, d, device, temperature, n_samples):
    """集成采样：K 模型逐步概率平均（结构与 m38.sample_chain 一致）"""
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    lig = torch.zeros(1, 0, 21, device=device)
    ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)

    n = int(n_samples)
    K = len(models)
    # 每模型编码器输出沿 batch 展开到 n 份
    enc_list = []
    for m in models:
        padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
        eo = m.forward_embedding(coords, ss, lig, ligm, padding_mask, conf)
        eo = {
            "encoder_out": [eo["encoder_out"][0].repeat(1, n, 1)],
            "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(n, 1)],
            "encoder_embedding": [],
            "encoder_states": [],
        }
        enc_list.append(eo)
    bos = alphabet.get_idx("<cath>")
    pred = torch.zeros(n, L + 1, dtype=torch.long, device=device)
    pred[:, 0] = bos
    states = [{} for _ in range(K)]
    for i in range(L):
        prev = pred[:, :i + 1]
        probs_sum = None
        for m, eo, st in zip(models, enc_list, states):
            logits, _ = m.decoder(prev, encoder_out=eo, incremental_state=st)
            logits = logits[:, :, -1]
            probs = F.softmax(logits / temperature, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        sample = torch.multinomial(probs_sum / K, 1).squeeze(-1)
        pred[:, i + 1] = sample
    return [''.join(alphabet.get_tok(t) for t in row.tolist()) for row in pred[:, 1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', type=str, required=True, help="逗号分隔 ckpt 路径/名")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--quick', type=int, default=0)
    parser.add_argument('--out', type=str, default="metrics_PACo_ens.json")
    parser.add_argument('--splits_v2', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    device = 0
    torch.manual_seed(args.seed)

    names = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    models = []
    for n in names:
        p = n if os.path.exists(n) else os.path.join(
            "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/PACoAptamer", n + ".pth")
        m, alphabet = m38.load_model(p, device)
        models.append(m)
    print(f"集成 {len(models)} 模型: {names}", flush=True)

    sp2 = json.load(open(os.path.join(m38.DATA_DIR, "splits_v2", "splits_v2.json")))
    test_stems = list(sp2["test"])
    data = load_s1(set(test_stems), cache=os.path.join(m38.DATA_DIR, "paco_s1_v2.pt"))
    test_stems = [s for s in test_stems if s in data]
    print(f"test 链 {len(test_stems)}", flush=True)

    n_s = args.quick or m38.N_SAMPLES
    results = {}
    t0 = time.time()
    for i, stem in enumerate(test_stems):
        d = data[stem]
        if len(d['seq']) > m38.MAX_LEN:
            continue
        native = d['seq']
        pool = []
        for t in m38.TEMPS:
            pool.extend(sample_chain_ens(models, alphabet, d, device, t, n_s))
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        results[stem] = {
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": float(np.mean(recs[:n_s])),
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
        "temps": m38.TEMPS,
        "n_samples_per_temp": n_s,
        "ensemble": names,
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(m38.DATA_DIR, "survey", args.out), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== PACoAptamer 集成（{'+'.join(names)}） =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
