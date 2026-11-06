#!/usr/bin/env python3
"""第 73 步：集成配置 val 选择（ens2 vs ens3 等）——避免在 test 上选配置

- val = splits_v2 val 96 链（paco_s1_v2.pt 缓存特征）
- 口径：t=0.1 × 20 样本均值 + 3 温度×20 池内 best（轻量排序用）
用法：CUDA_VISIBLE_DEVICES=1 python data/73_val_select_ensemble.py --gpu 0 \
    --configs ens2:s2_v2_seed43,s2_v2_ssdo15 ens3:s2_v2_batch4r,s2_v2_seed43,s2_v2_ssdo15
"""
import os, sys, json, argparse
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

import importlib.util
spec39 = importlib.util.spec_from_file_location("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
m39 = importlib.util.module_from_spec(spec39); spec39.loader.exec_module(m39)
spec38 = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
m38 = importlib.util.module_from_spec(spec38); spec38.loader.exec_module(m38)
from PACoAptamer.data_utils import load_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--configs", nargs="+", required=True, help="name:ckpt1,ckpt2 [...]")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(42)

    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    val_stems = list(sp["val"])
    data = load_s1(set(val_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    val_stems = [s for s in val_stems if s in data]
    print(f"val {len(val_stems)} 链", flush=True)

    out = {}
    for cfg in args.configs:
        name, ckpts = cfg.split(":", 1)
        models = []
        for n in ckpts.split(","):
            p = n if os.path.exists(n) else os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
            m, alphabet = m38.load_model(p, device)
            models.append(m)
        t01_recs, best_recs = [], []
        for stem in val_stems:
            d = data[stem]
            if len(d["seq"]) > m38.MAX_LEN:
                continue
            native = d["seq"]
            pool01 = m39.sample_chain_ens(models, alphabet, d, device, 0.1, args.n)
            recs01 = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool01]
            t01_recs.append(float(np.mean(recs01)))
            pool_all = list(pool01)
            for t in (0.5, 1.0):
                pool_all.extend(m39.sample_chain_ens(models, alphabet, d, device, t, args.n))
            recs_all = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool_all]
            best_recs.append(max(recs_all))
        out[name] = {"val_t01_mean": float(np.mean(t01_recs)), "val_best_mean": float(np.mean(best_recs)),
                     "n": len(t01_recs)}
        print(f"{name}: t01 {out[name]['val_t01_mean']:.4f} | best {out[name]['val_best_mean']:.4f}", flush=True)

    json.dump(out, open(os.path.join(SUR, "73_val_select_ensemble.json"), "w"), indent=1)
    print("写入 survey/73_val_select_ensemble.json", flush=True)


if __name__ == "__main__":
    main()
