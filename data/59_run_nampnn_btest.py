#!/usr/bin/env python3
"""第 59 步：NA-MPNN B 层 v2 test 推理（重训权重 × canonical 477 链）

- 输入：data/canonical_v2/{stem}.pdb（51 号权威库）
- 权重：models/NA-MPNN_btrain_v2/s_{step}.pt（54 号 v2 重训，按 val 选优后传入）
- 口径：t=0.1 × 10 样本（与 A 层一致）；batch_size=10 单次推理出 10 样本（提速，官方支持）
- 输出：survey/nampnn_out_btest_v2/{stem}.seqs
用法：python data/59_run_nampnn_btest.py --gpu 0 --ckpt <path> [--taskfile x.json]
"""
import os, sys, json, subprocess, shutil, argparse
from concurrent.futures import ThreadPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
NAMPNN_DIR = os.path.join(ROOT, "tools", "NA-MPNN")
IN_DIR = os.environ.get("NP_IN_DIR", os.path.join(DATA_DIR, "canonical_v2"))
OUT_DIR = os.environ.get("NP_OUT_DIR", os.path.join(DATA_DIR, "survey", "nampnn_out_btest_v2"))
PY = "/root/miniconda3/envs/aptamer_eval/bin/python"

RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}


def run_one(stem, gpu, ckpt):
    out_f = os.path.join(OUT_DIR, f"{stem}.seqs")
    if os.path.exists(out_f) and os.path.getsize(out_f) > 100:
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    n_atom = sum(1 for l in open(pdb_path) if l.startswith("ATOM"))
    if n_atom == 0:
        return (stem, "no_coords")
    workdir = os.path.join(OUT_DIR, f"tmp_{stem}")
    cmd = [PY, "./inference/run.py",
           "--model_type", "na_mpnn",
           "--mode", "design",
           "--checkpoint_na_mpnn", ckpt,
           "--batch_size", "10",
           "--pdb_path", pdb_path,
           "--out_folder", workdir,
           "--output_pdbs", "0"]
    try:
        r = subprocess.run(cmd, cwd=NAMPNN_DIR, capture_output=True,
                           timeout=900, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return (stem, "timeout")
    fa = os.path.join(workdir, "seqs", f"{stem}.fa")
    if r.returncode != 0 or not os.path.exists(fa):
        print(f"    [诊断] {stem} rc={r.returncode} stderr={r.stderr.decode()[-300:]!r}", flush=True)
        shutil.rmtree(workdir, ignore_errors=True)
        return (stem, "failed")
    lines = [l.strip() for l in open(fa) if l.strip()]
    seqs = []
    for j, line in enumerate(lines):
        if line.startswith(">") and "id=" in line and j + 1 < len(lines):
            seq = "".join(RNA_MAP.get(c, "") for c in lines[j + 1])
            if len(seq) > 0:
                seqs.append(seq)
    shutil.rmtree(workdir, ignore_errors=True)
    if not seqs:
        return (stem, "no_output")
    with open(out_f, "w") as f:
        f.write("\n".join(seqs) + "\n")
    return (stem, f"ok_{len(seqs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--taskfile", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.taskfile:
        tasks = json.load(open(args.taskfile))
    else:
        sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
        tasks = sorted(sp["test"])
    print(f"[NAMPNN-btest GPU{args.gpu}] {len(tasks)} 条链, ckpt={os.path.basename(args.ckpt)}", flush=True)
    stats = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, stem, args.gpu, args.ckpt): stem for stem in tasks}
        for i, fut in enumerate(futs):
            stem, status = fut.result()
            k = status.split("_")[0]
            stats[k] = stats.get(k, 0) + 1
            if (i + 1) % 25 == 0:
                print(f"进度 {i+1}/{len(tasks)} {stats}", flush=True)
    print(f"[NAMPNN-btest GPU{args.gpu}] 完成: {stats}", flush=True)
