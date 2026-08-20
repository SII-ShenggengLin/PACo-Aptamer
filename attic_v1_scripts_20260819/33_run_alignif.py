#!/usr/bin/env python3
"""第 33 步：AlignIF 批量推理（官方权重 × 评估集，A 层）

- 模型：tools/AlignIF/ckpts/alignif-0.ckpt（从 Google Drive ckpts.zip 解压）
- 输入：grnade_inputs/{stem}.pdb（链级 PDB，与 gRNAde/NA-MPNN 同输入）
- 输出：survey/alignif_out/{stem}/out.fasta（官方 inference.py 格式）
- 采样：n_samples=20, temperature=0.1（统一口径）
- use_msta=False：避免 MStA 检索引入训练期结构信息（与 gRNAde/RhoDesign 公平对比）
- GPU：1、2（CUDA_VISIBLE_DEVICES 限制后设备号从 0 数）
"""
import os, sys, subprocess, shutil, argparse, json

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
ALIGNIF_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/AlignIF"
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs")
OUT_DIR = os.path.join(DATA_DIR, "survey", "alignif_out")
CKPT = os.path.join(ALIGNIF_DIR, "ckpts", "alignif-0.ckpt")
PY = "/root/miniconda3/envs/aptamer_eval/bin/python"
# importlib 模块名是 'rnet-inference.src.utils' → 需 tools/ 在 PYTHONPATH
PYTHONPATH = os.path.dirname(ALIGNIF_DIR)

def run_one(stem, gpu):
    out_sub = os.path.join(OUT_DIR, stem)
    if os.path.exists(os.path.join(out_sub, "out.fasta")):
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    if not os.path.exists(pdb_path):
        return (stem, "no_pdb")
    cmd = [PY, "inference.py",
           "--pdbs_dir", IN_DIR,
           "--checkpoint", CKPT,
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
        print(f"    [诊断] {stem} rc={r.returncode} {err}", flush=True)
        return (stem, "failed")
    return (stem, "ok")

def main(gpu, stems):
    print(f"[AlignIF GPU{gpu}] {len(stems)} 链", flush=True)
    stats = {}
    for i, stem in enumerate(stems):
        s, st = run_one(stem, gpu)
        stats[st] = stats.get(st, 0) + 1
        print(f"  {s}: {st} [{i+1}/{len(stems)}] {stats}", flush=True)
    print(f"[AlignIF GPU{gpu}] 完成: {stats}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--taskfile", type=str, required=True)
    args = parser.parse_args()
    main(args.gpu, json.load(open(args.taskfile)))
