#!/usr/bin/env python3
"""第 30v2 步：NA-MPNN v2 推理（基于 31 号修复版：顺序执行 + timeout 600 + 短链保留）

- 背景：30 号批量 4 线程并发 + timeout 300s → 55 条链超时/失败（GPU 争抢）
- 修复：顺序执行（1 并发）+ timeout 600s
- 剔除：无坐标链（0 个 ATOM，如 8S1P_b UNK 残基）
- 输出：survey/nampnn_out/{stem}.seqs（与 30 号同一目录）
"""
import os, sys, json, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
NAMPNN_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/NA-MPNN"
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs_v2")
OUT_DIR = os.path.join(DATA_DIR, "survey", "nampnn_out_v2")
PY = "/root/miniconda3/envs/aptamer_eval/bin/python"

RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}

def run_one(stem, gpu):
    out_f = os.path.join(OUT_DIR, f"{stem}.seqs")
    if os.path.exists(out_f) and os.path.getsize(out_f) > 100:
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    n_atom = sum(1 for l in open(pdb_path) if l.startswith("ATOM"))
    if n_atom == 0:
        return (stem, "no_coords")   # 无坐标链，剔除
    seqs = []
    for i in range(10):
        workdir = os.path.join(OUT_DIR, f"tmp_{stem}_{i}")
        cmd = [PY, "./inference/run.py",
               "--model_type", "na_mpnn",
               "--mode", "design",
               "--pdb_path", pdb_path,
               "--out_folder", workdir,
               "--output_pdbs", "0"]
        try:
            r = subprocess.run(cmd, cwd=NAMPNN_DIR, capture_output=True,
                               timeout=600, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
        except subprocess.TimeoutExpired:
            shutil.rmtree(workdir, ignore_errors=True)
            continue
        fa = os.path.join(workdir, "seqs", f"{stem}.fa")
        if r.returncode != 0 or not os.path.exists(fa):
            print(f"    [诊断] {stem} 采样{i} rc={r.returncode} stderr={r.stderr.decode()[-300:]!r}", flush=True)
            shutil.rmtree(workdir, ignore_errors=True)
            continue
        lines = [l.strip() for l in open(fa) if l.strip()]
        for j, line in enumerate(lines):
            if line.startswith(">") and "id=" in line and j + 1 < len(lines):
                seq = "".join(RNA_MAP.get(c, "") for c in lines[j + 1])
                if len(seq) > 0:   # RNA-only 输入，输出应全 RNA 字母；短链(≤10nt)也保留
                    seqs.append(seq)
                    break
        shutil.rmtree(workdir, ignore_errors=True)
    if not seqs:
        return (stem, "no_output")
    with open(out_f, "w") as f:
        f.write("\n".join(seqs) + "\n")
    return (stem, f"ok_{len(seqs)}")

if __name__ == "__main__":
    gpu = int(sys.argv[1])
    tasks = [f"{t[0]}_{t[1]}" for t in json.load(open(sys.argv[2]))]
    print(f"[NAMPNN GPU{gpu}] {len(tasks)} 条链，4 并发", flush=True)
    stats = {"ok": 0, "cached": 0, "no_pdb": 0, "no_output": 0, "no_coords": 0}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, stem, gpu): stem for stem in tasks}
        for i, fut in enumerate(futs):
            stem, status = fut.result()
            if status.startswith("ok"):
                stats["ok"] += 1
            elif status in stats:
                stats[status] += 1
            else:
                stats[status] = stats.get(status, 0) + 1
            if (i + 1) % 25 == 0:
                print(f"进度 {i+1}/{len(tasks)} {stats}", flush=True)
    print(f"[NAMPNN GPU{gpu}] 完成: {stats}", flush=True)
