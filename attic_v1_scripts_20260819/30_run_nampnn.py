#!/usr/bin/env python3
"""第 30 步：NA-MPNN 批量推理（官方权重 × 评估集）

- 模型：baker-laboratory/NA-MPNN 官方权重 s_19137.pt（仓库自带）
- 输入：grnade_inputs/{stem}.pdb（链级 PDB）
- 输出：survey/nampnn_out/{stem}.seqs（每行一条设计序列，字母映射 b=A/d=C/h=G/u=U）
- 采样：t=0.1 × 10 次（不同随机种子）
- GPU：1、2
"""
import os, sys, json, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
NAMPNN_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/NA-MPNN"
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs")
OUT_DIR = os.path.join(DATA_DIR, "survey", "nampnn_out")
os.makedirs(OUT_DIR, exist_ok=True)
PY = "/root/miniconda3/envs/aptamer_eval/bin/python"

# RNA 单字母映射（README：b=A, d=C, h=G, u=U；RNA 设计输出只有这些小写）
RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}

def run_one(task, gpu):
    stem = f"{task[0]}_{task[1]}"   # (pdb, chain) → stem
    out_f = os.path.join(OUT_DIR, f"{stem}.seqs")
    if os.path.exists(out_f) and os.path.getsize(out_f) > 100:
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    if not os.path.exists(pdb_path):
        return (stem, "no_pdb")
    seqs = []
    for i in range(10):   # 10 次采样（随机种子）
        workdir = os.path.join(OUT_DIR, f"tmp_{stem}_{i}")
        cmd = [PY, "./inference/run.py",
               "--model_type", "na_mpnn",
               "--mode", "design",
               "--pdb_path", pdb_path,
               "--out_folder", workdir,
               "--output_pdbs", "0"]
        r = subprocess.run(cmd, cwd=NAMPNN_DIR, capture_output=True, timeout=300)
        fa = os.path.join(workdir, "seqs", f"{os.path.basename(stem)}.fa")
        if r.returncode != 0 or not os.path.exists(fa):
            continue
        # 解析 fasta：id=1 的设计序列行
        lines = [l.strip() for l in open(fa) if l.strip()]
        for j, line in enumerate(lines):
            if line.startswith(">") and "id=" in line and j + 1 < len(lines):
                seq_raw = lines[j + 1]
                # 字母映射：只保留 RNA 字符（去掉蛋白部分——输入 RNA-only，输出应全 RNA）
                seq = "".join(RNA_MAP.get(c, "") for c in seq_raw)
                if len(seq) > 10:
                    seqs.append(seq)
                    break
        # 清理
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    if not seqs:
        return (stem, "no_output")
    with open(out_f, "w") as f:
        f.write("\n".join(seqs) + "\n")
    return (stem, f"ok_{len(seqs)}")

def main(gpu, task_list):
    stats = {"ok": 0, "cached": 0, "no_pdb": 0, "no_output": 0}
    with ThreadPoolExecutor(max_workers=4) as ex:   # GPU 上并行 4 个子进程
        futs = {ex.submit(run_one, stem, gpu): stem for stem in task_list}
        for i, fut in enumerate(futs):
            stem, status = fut.result()
            if status.startswith("ok"):
                stats["ok"] += 1
            elif status == "cached":
                stats["cached"] += 1
            elif status == "no_output":
                stats["no_output"] += 1
                print(f"  [FAIL] {stem}: 无输出", flush=True)
            if (i + 1) % 25 == 0:
                print(f"进度 {i+1}/{len(task_list)} {stats}", flush=True)
    print(f"[GPU{gpu}] 完成: {stats}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--taskfile", type=str, required=True)
    args = parser.parse_args()
    main(args.gpu, [tuple(t) for t in json.load(open(args.taskfile))])
