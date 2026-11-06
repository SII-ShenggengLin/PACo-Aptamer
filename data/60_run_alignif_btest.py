#!/usr/bin/env python3
"""第 60 步：AlignIF B 层 v2 test 推理（重训权重 × canonical 477 链）

- 输入：data/canonical_v2/{stem}.pdb
- 权重：models/AlignIF_btrain_v2/alignif-seed=0-epoch=XX.ckpt（v2 无 MStA 重训，best val/loss）
- 口径：n_samples=20, temperature=0.1, use_msta=False（与 A 层 v2 完全一致）
- 输出：survey/alignif_out_btest_v2/{stem}/out.fasta
用法：python data/60_run_alignif_btest.py --gpu 0 --ckpt <path> [--taskfile x.json]
"""
import os, sys, subprocess, argparse, json

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
ALIGNIF_DIR = os.path.join(ROOT, "tools", "AlignIF")
IN_DIR = os.environ.get("AI_IN_DIR", os.path.join(DATA_DIR, "canonical_v2"))
OUT_DIR = os.environ.get("AI_OUT_DIR", os.path.join(DATA_DIR, "survey", "alignif_out_btest_v2"))
PY = "/root/miniconda3/envs/aptamer_eval/bin/python"
PYTHONPATH = os.path.dirname(ALIGNIF_DIR)


def run_one(stem, gpu, ckpt):
    out_sub = os.path.join(OUT_DIR, stem)
    if os.path.exists(os.path.join(out_sub, "out.fasta")):
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    if not os.path.exists(pdb_path):
        return (stem, "no_pdb")
    cmd = [PY, "inference.py",
           "--pdbs_dir", IN_DIR,
           "--checkpoint", ckpt,
           "--n_samples", "20",
           "--temperature", "0.1",
           "--use_msta", "False",
           "--output_dir", OUT_DIR,
           "--input_structure_path", pdb_path]
    env = {**os.environ, "PYTHONPATH": PYTHONPATH,
           "CUDA_VISIBLE_DEVICES": str(gpu), "NUPACKHOME": "/tmp"}
    try:
        r = subprocess.run(cmd, cwd=ALIGNIF_DIR, capture_output=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return (stem, "timeout")
    if r.returncode != 0 or not os.path.exists(os.path.join(out_sub, "out.fasta")):
        err = r.stderr.decode()[-500:] if r.stderr else ""
        print(f"    [诊断] {stem} rc={rc if (rc:=r.returncode) else ''} {err}", flush=True)
        return (stem, "failed")
    return (stem, "ok")


def main(gpu, ckpt, stems):
    print(f"[AlignIF-btest GPU{gpu}] {len(stems)} 链, ckpt={os.path.basename(ckpt)}", flush=True)
    stats = {}
    for i, stem in enumerate(stems):
        s, st = run_one(stem, gpu, ckpt)
        stats[st] = stats.get(st, 0) + 1
        if (i + 1) % 10 == 0 or i == len(stems) - 1:
            print(f"  进度 [{i+1}/{len(stems)}] {stats}", flush=True)
    print(f"[AlignIF-btest GPU{gpu}] 完成: {stats}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--taskfile", type=str, default=None)
    args = parser.parse_args()
    if args.taskfile:
        stems = json.load(open(args.taskfile))
    else:
        sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
        stems = sorted(sp["test"])
    main(args.gpu, args.ckpt, stems)
