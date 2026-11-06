#!/usr/bin/env python3
"""第 71 步：PACoAptamer 集成 A 层评估（indep 235 ∪ zs 93 = 289 链）

- 与 68 号同输入口径（grnade_inputs_v2 → PACo 9 原子 + RNAfold SS，native=manifest）
- 采样与 39 号同：K 模型逐步概率平均，3 温度 × 100
- 输出：survey/metrics_PACo_av2_ens{K}.json + paco_av2_pool_ens{K}/{stem}.json（pool 留档，供 69 号）
用法：CUDA_VISIBLE_DEVICES=2 python data/71_eval_paco_ensemble_av2.py \
    --ckpts s2_v2_seed43,s2_v2_ssdo15 --gpu 0
"""
import os, sys, json, argparse, time
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

import importlib.util
spec68 = importlib.util.spec_from_file_location("m68", os.path.join(DATA_DIR, "68_eval_paco_av2.py"))
m68 = importlib.util.module_from_spec(spec68)
spec68.loader.exec_module(m68)   # build_feature（含 RNAfold SS）
spec39 = importlib.util.spec_from_file_location("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
m39 = importlib.util.module_from_spec(spec39)
spec39.loader.exec_module(m39)   # sample_chain_ens
spec38 = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
m38 = importlib.util.module_from_spec(spec38)
spec38.loader.exec_module(m38)   # load_model / TEMPS / N_SAMPLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", type=str, required=True)
    ap.add_argument("--taskfile", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None, help="输出后缀（默认 ens{K}）")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    names = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    models = []
    for n in names:
        p = n if os.path.exists(n) else os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m38.load_model(p, device)
        models.append(m)
    tag = args.tag or f"ens{len(models)}"

    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    stems = sorted(set(sp["independent_test"]) | set(sp["zero_shot_targets"]))
    if args.taskfile:
        stems = json.load(open(args.taskfile))

    out_dir = os.path.join(SUR, f"paco_av2_pool_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    results, skipped = {}, {}
    t0 = time.time()
    for i, stem in enumerate(stems):
        pf = os.path.join(out_dir, f"{stem}.json")
        if os.path.exists(pf):
            d = json.load(open(pf))
            results[stem] = {k: d[k] for k in ("len", "recovery_best", "recovery_t01_mean", "n_pool")}
            continue
        d, status = m68.build_feature(stem)
        if d is None:
            skipped[stem] = status
            print(f"  [SKIP] {stem}: {status}", flush=True)
            continue
        pool = []
        for t in m38.TEMPS:
            pool.extend(m39.sample_chain_ens(models, alphabet, d, device, t, m38.N_SAMPLES))
        native = d["seq"]
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        rec = {"len": len(native), "recovery_best": max(recs),
               "recovery_t01_mean": float(np.mean(recs[:m38.N_SAMPLES])), "n_pool": len(pool)}
        results[stem] = rec
        json.dump({"native": native, "pool": pool, **rec}, open(pf, "w"))
        if (i + 1) % 25 == 0:
            print(f"进度 {i+1}/{len(stems)} | {time.time()-t0:.0f}s", flush=True)

    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(stems)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])),
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])),
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])),
        "skipped": skipped,
        "note": f"PACo 集成 A 层 v2（{'+'.join(names)} × indep∪zs 289 链；native=manifest 口径 grnade_out_v2）",
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(SUR, f"metrics_PACo_av2_{tag}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("skipped", "note")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
