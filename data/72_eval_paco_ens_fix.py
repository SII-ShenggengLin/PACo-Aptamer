#!/usr/bin/env python3
"""第 72 步：PACo 集成 B 层 canonical 修复池（40 条 native 不一致链重采样）

- 与 56 号同逻辑（canonical_v2 特征重建 + canonical native），采样器换 39 号集成
- 输出 survey/paco_fix_pool_ens{K}/{stem}.json；配合 63 号 PACO_METRICS/PACO_FIXPOOL 重建主表
用法：CUDA_VISIBLE_DEVICES=2 python data/72_eval_paco_ens_fix.py --ckpts a,b --gpu 0
"""
import os, sys, json, argparse
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
CANON = os.path.join(DATA_DIR, "canonical_v2")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

import importlib.util
spec56 = importlib.util.spec_from_file_location("m56", os.path.join(DATA_DIR, "56_eval_paco_fix.py"))
m56 = importlib.util.module_from_spec(spec56)
spec56.loader.exec_module(m56)   # build_feature（canonical 版）
spec39 = importlib.util.spec_from_file_location("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
m39 = importlib.util.module_from_spec(spec39)
spec39.loader.exec_module(m39)
spec38 = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
m38 = importlib.util.module_from_spec(spec38)
spec38.loader.exec_module(m38)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", type=str, required=True)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    names = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    models = []
    for n in names:
        p = n if os.path.exists(n) else os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m38.load_model(p, device)
        models.append(m)
    tag = args.tag or f"ens{len(models)}"

    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    stems = json.load(open(os.path.join(SUR, "55_repair_manifest.json")))["paco_btest"]
    out_dir = os.path.join(SUR, f"paco_fix_pool_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    for i, stem in enumerate(stems):
        pf = os.path.join(out_dir, f"{stem}.json")
        if os.path.exists(pf):
            continue
        native = canon[stem]["seq"]
        d = m56.build_feature(stem, native)
        pool = []
        for t in m38.TEMPS:
            pool.extend(m39.sample_chain_ens(models, alphabet, d, device, t, m38.N_SAMPLES))
        npos = [j for j, c in enumerate(native) if c != "N"]
        recs = [sum(1 for j in npos if j < len(s) and s[j] == native[j]) / len(npos)
                for s in pool if len(s) == len(native)]
        rec = {"len": len(native), "recovery_best": max(recs),
               "recovery_t01_mean": float(np.mean(recs[:m38.N_SAMPLES])), "n_pool": len(pool)}
        json.dump({"native": native, "pool": pool, **rec}, open(pf, "w"))
        print(f"[{i+1}/{len(stems)}] {stem}: best {rec['recovery_best']:.3f}", flush=True)
    print(f"paco_fix_pool_{tag} 完成（{len(stems)} 链）", flush=True)


if __name__ == "__main__":
    main()
