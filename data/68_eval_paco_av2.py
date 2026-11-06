#!/usr/bin/env python3
"""第 68 步：PACo A 层 v2 重训权重评估（indep 235 ∪ zero_shot 93 = 289 链）

- 权重：models/PACoAptamer/s2_v2.pth（与 B 层主表同权重；A 层两层同权重是"PACo 无官方权重"的如实口径）
- 输入：grnade_inputs_v2/{stem}.pdb（43 号管线，A 层既定输入）→ PACo 9 原子特征 + RNAfold SS
  （ATOM+HETATM 保留修饰残基并映射母碱基，不丢残基——比 46 号案例脚本更严）
- native：grnade_out_v2/{stem}.native（manifest 口径，与其他方法 A 层行一致）
- 采样：3 温度 × 100（38 号同函数同口径）
- 输出：survey/paco_av2_pool/{stem}.json（pool 留档）+ 直接并入 survey/metrics_PACo_av2.json
用法：python data/68_eval_paco_av2.py --gpu 0 [--taskfile x.json]
"""
import os, sys, json, argparse, subprocess, re
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
IN_DIR = os.path.join(SUR, "grnade_inputs_v2")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from PACoAptamer.data_utils import PACO9, ss_to_pairs

import importlib.util
spec38 = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
m38 = importlib.util.module_from_spec(spec38)
spec38.loader.exec_module(m38)
spec32 = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec32)
spec32.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}

RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"


def build_feature(stem):
    """grnade_inputs_v2 PDB → (coords9, ss_pairs)；序列真值用 grnade_out_v2 native"""
    native_p = os.path.join(SUR, "grnade_out_v2", f"{stem}.native")
    if not os.path.exists(native_p):
        return None, "no_native"
    native = open(native_p).read().strip()
    residues = {}
    order = []
    with open(os.path.join(IN_DIR, f"{stem}.pdb")) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            if rn not in STD and rn not in MOD_MAP:
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip().replace("*", "'")
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key][atom] = xyz
    L = len(order)
    if L != len(native):
        return None, f"len_mismatch pdb={L} native={len(native)}"
    coords9 = np.full((L, 9, 3), np.nan, dtype=np.float32)
    for i, key in enumerate(order):
        for j, atom in enumerate(PACO9):
            if atom in residues[key]:
                coords9[i, j] = residues[key][atom]
    r = subprocess.run([RNAFOLD, "--noPS"], input=native + "\n", capture_output=True, text=True, timeout=60)
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    ss = m[0] if m else "." * len(native)
    return {"coords9": coords9, "ss_pairs": ss_to_pairs(ss), "seq": native}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--taskfile", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "models", "PACoAptamer", "s2_v2.pth"))
    args = ap.parse_args()

    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    stems = sorted(set(sp["independent_test"]) | set(sp["zero_shot_targets"]))
    if args.taskfile:
        stems = json.load(open(args.taskfile))
    device = torch.device(f"cuda:{args.gpu}")
    model, alphabet = m38.load_model(args.ckpt, device)

    # 【2026-08-17 补丁】pool 目录/metrics 输出支持环境变量覆盖——多 ckpt 并行复评时
    # 防缓存误用（旧 pool 会把别的 ckpt 的采样当命中）与 metrics 相互覆盖
    out_dir = os.environ.get("PACO_AV2_POOL", os.path.join(SUR, "paco_av2_pool"))
    out_json = os.environ.get("PACO_AV2_OUT", os.path.join(SUR, "metrics_PACo_av2.json"))
    os.makedirs(out_dir, exist_ok=True)
    n_s = m38.N_SAMPLES
    results = {}
    skipped = {}
    for i, stem in enumerate(stems):
        pf = os.path.join(out_dir, f"{stem}.json")
        if os.path.exists(pf):
            d = json.load(open(pf))
            results[stem] = {"len": d["len"], "recovery_best": d["recovery_best"],
                             "recovery_t01_mean": d["recovery_t01_mean"], "n_pool": len(d["pool"])}
            continue
        d, status = build_feature(stem)
        if d is None:
            skipped[stem] = status
            print(f"  [SKIP] {stem}: {status}", flush=True)
            continue
        pool = []
        for t in m38.TEMPS:
            pool.extend(m38.sample_chain(model, alphabet, d, device, t, n_s, False))
        native = d["seq"]
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        rec = {"len": len(native), "recovery_best": max(recs),
               "recovery_t01_mean": float(np.mean(recs[:n_s])), "n_pool": len(pool)}
        results[stem] = rec
        json.dump({"native": native, "pool": pool, **rec},
                  open(pf, "w"))
        if (i + 1) % 25 == 0:
            print(f"进度 {i+1}/{len(stems)}", flush=True)

    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(stems)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])),
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])),
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])),
        "skipped": skipped,
        "note": f"PACo A 层 v2（{os.path.basename(args.ckpt)} 重训权重 × indep∪zs 289 链；native=manifest 口径 grnade_out_v2）",
    }
    json.dump({"summary": summary, "per_chain": results},
              open(out_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("skipped", "note")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
