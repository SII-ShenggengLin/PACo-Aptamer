#!/usr/bin/env python3
"""第 56 步：PACo B 层 test 40 条 native 不一致链的重评估（canonical 真值）

- 从 canonical_v2/{stem}.pdb 重建特征（PACo 9 原子 + RNAfold SS → ss_pairs + canonical seq）
- 复用 38 号的 load_model/sample_chain（s2_v2.pth，3 温度 × 100，与主表同口径）
- 恢复率对 canonical native；结果并入 metrics_PACo_v2_main.json（per_chain 替换 + summary 重算）
- 输出另存 survey/paco_fix_pool/{stem}.json（样本池留档）

用法：python data/56_eval_paco_fix.py --gpu 2
"""
import os, sys, json, argparse, subprocess
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
CANON = os.path.join(DATA_DIR, "canonical_v2")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from PACoAptamer.data_utils import PACO9, ss_to_pairs

import importlib.util
spec = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
m38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m38)   # 复用 load_model / sample_chain / TEMPS / N_SAMPLES

RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"


def build_feature(stem, seq):
    """canonical PDB → coords9 (L,9,3) + ss_pairs (L,L) + seq（与 45 号管线同约定）"""
    residues = {}
    order = []
    with open(os.path.join(CANON, f"{stem}.pdb")) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key][atom] = xyz
    L = len(order)
    assert L == len(seq), f"{stem}: 残基数 {L} != 序列 {len(seq)}"
    coords9 = np.full((L, 9, 3), np.nan, dtype=np.float32)
    for i, key in enumerate(order):
        for j, atom in enumerate(PACO9):
            if atom in residues[key]:
                coords9[i, j] = residues[key][atom]
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n", capture_output=True, text=True)
    ss = r.stdout.strip().split("\n")[-1].split()[0]
    return {"coords9": coords9, "ss_pairs": ss_to_pairs(ss), "seq": seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "models", "PACoAptamer", "s2_v2.pth"))
    ap.add_argument("--taskfile", type=str, default=None)   # 指定链清单则只跑这些（默认 = 55 号清单 40 条）
    args = ap.parse_args()

    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    if args.taskfile:
        stems = json.load(open(args.taskfile))
    else:
        stems = json.load(open(os.path.join(SUR, "55_repair_manifest.json")))["paco_btest"]

    device = torch.device(f"cuda:{args.gpu}")
    model, alphabet = m38.load_model(args.ckpt, device)

    # 【2026-08-17 补丁】修复池目录支持环境变量覆盖（换 ckpt 时防覆盖旧池/缓存误用）
    out_dir = os.environ.get("PACO_FIXPOOL", os.path.join(SUR, "paco_fix_pool"))
    os.makedirs(out_dir, exist_ok=True)
    n_s = m38.N_SAMPLES
    results = {}
    for i, stem in enumerate(stems):
        seq = canon[stem]["seq"]
        d = build_feature(stem, seq)
        pool = []
        for t in m38.TEMPS:
            pool.extend(m38.sample_chain(model, alphabet, d, device, t, n_s, False))
        recs = [sum(1 for a, b in zip(s, seq) if a == b) / len(seq) for s in pool]
        results[stem] = {
            "len": len(seq),
            "recovery_best": max(recs),
            "recovery_t01_mean": float(np.mean(recs[:n_s])),
            "n_pool": len(pool),
        }
        json.dump({"native": seq, "pool": pool},
                  open(os.path.join(out_dir, f"{stem}.json"), "w"))
        print(f"  {stem}: len={len(seq)} best={max(recs):.3f} [{i+1}/{len(stems)}]", flush=True)

    # 并入 metrics_PACo_v2_main.json
    mp = os.path.join(SUR, "metrics_PACo_v2_main.json")
    m = json.load(open(mp))
    m["per_chain"].update(results)
    pc = m["per_chain"]
    m["summary"]["n_chains"] = len(pc)
    m["summary"]["recovery_best_mean"] = float(np.mean([r["recovery_best"] for r in pc.values()]))
    m["summary"]["recovery_best_median"] = float(np.median([r["recovery_best"] for r in pc.values()]))
    m["summary"]["recovery_t01_mean_mean"] = float(np.mean([r["recovery_t01_mean"] for r in pc.values()]))
    m["summary"]["canonical_fix_note"] = (f"{len(results)} 条链（含 3 条多拷贝拼接修正）按 canonical_v2 "
                                          "真值重评估（56 号），覆盖旧 s2 拼接/口径值")
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
    print(f"\n并入后: n={len(pc)} best_mean={m['summary']['recovery_best_mean']:.4f} "
          f"t01_mean={m['summary']['recovery_t01_mean_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
