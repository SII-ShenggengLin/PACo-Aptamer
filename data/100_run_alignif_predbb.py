#!/usr/bin/env python3
"""第 100 步：AlignIF 在 RhoFold+ 预测骨架上的批量推理（86_sctm 193 链）

- 输入：survey/100_predbb_inputs/{stem}.pdb → 符号链接到 86_sctm/native_{stem}.pdb
  （AlignIF 按输入文件 basename 命名输出，链接名不带 native_ 前缀）
- 权重：models/AlignIF_btrain_v2/alignif-seed=0-epoch=79.ckpt（60 号同款）
- 口径：n_samples=20, temperature=0.1, use_msta=False，单链 timeout 1800s
- 输出：survey/alignif_out_predbb/{stem}/out.fasta
- 断点续跑：已有 out.fasta 的链跳过
- phase run     并行推理（ThreadPoolExecutor 管理子进程，workers 默认 6）
- phase metrics 仿 62 号算逐链 best/t01（native=canonical_v2.json，N/- 位跳过）
  → survey/100_alignif_predbb.json（先写 .tmp 再 rename）

环境（2026-10-16 起）：aptamer_eval 环境丢失，改用 odesign_service_env
（python 3.10.20 + torch 2.3.1+cu121 + 真实 torch_scatter/cluster），
lightning/arnie 装在 data/pylibs_alignif（--no-deps，不动 odesign 环境）。
用法：CUDA_VISIBLE_DEVICES=0 python data/100_run_alignif_predbb.py --phase run --workers 6
"""
import os, sys, json, argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
ALIGNIF_DIR = os.path.join(ROOT, "tools", "AlignIF")
SRC_DIR = os.path.join(SUR, "86_sctm")
IN_DIR = os.path.join(SUR, "100_predbb_inputs")
OUT_DIR = os.path.join(SUR, "alignif_out_predbb")
PY = "/root/miniconda3/envs/odesign_service_env/bin/python"
CKPT = os.path.join(ROOT, "models", "AlignIF_btrain_v2", "alignif-seed=0-epoch=79.ckpt")


def stems_193():
    d = json.load(open(os.path.join(SUR, "86_sctm.json")))
    return sorted(d["per_chain"].keys())


def phase_links():
    os.makedirs(IN_DIR, exist_ok=True)
    n_new = 0
    for stem in stems_193():
        src = os.path.join(SRC_DIR, f"native_{stem}.pdb")
        dst = os.path.join(IN_DIR, f"{stem}.pdb")
        if not os.path.exists(src):
            print(f"  [links 缺源] {stem}", flush=True)
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
        n_new += 1
    print(f"[links] 新建 {n_new}，总计 {len(os.listdir(IN_DIR))}", flush=True)


def run_one(stem):
    out_sub = os.path.join(OUT_DIR, stem)
    if os.path.exists(os.path.join(out_sub, "out.fasta")):
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    cmd = [PY, "inference.py",
           "--pdbs_dir", IN_DIR,
           "--checkpoint", CKPT,
           "--n_samples", "20",
           "--temperature", "0.1",
           "--use_msta", "False",
           "--output_dir", OUT_DIR,
           "--input_structure_path", pdb_path]
    env = {**os.environ,
           "PYTHONPATH": os.path.join(DATA_DIR, "pylibs_alignif") + ":" + os.path.join(ROOT, "tools"),
           "CUDA_VISIBLE_DEVICES": "0",
           "NUPACKHOME": "/tmp"}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ALIGNIF_DIR, capture_output=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return (stem, "timeout")
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(os.path.join(out_sub, "out.fasta")):
        err = r.stderr.decode()[-400:] if r.stderr else ""
        print(f"    [诊断] {stem} rc={r.returncode} {err}", flush=True)
        return (stem, "failed")
    return (stem, f"ok {dt:.0f}s")


def phase_run(workers):
    stems = stems_193()
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [s for s in stems
            if not os.path.exists(os.path.join(OUT_DIR, s, "out.fasta"))]
    print(f"[run] 共 {len(stems)} 链，待跑 {len(todo)}（workers={workers}）", flush=True)
    stats = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (stem, st) in enumerate(ex.map(run_one, todo)):
            key = st.split()[0]
            stats[key] = stats.get(key, 0) + 1
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                print(f"  进度 [{i+1}/{len(todo)}] {stats} | {time.time()-t0:.0f}s", flush=True)
    print(f"[run] 完成: {stats} | 总耗时 {time.time()-t0:.0f}s", flush=True)


def phase_metrics():
    import numpy as np
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    stems = stems_193()
    results, failed = {}, []
    for stem in stems:
        fa = os.path.join(OUT_DIR, stem, "out.fasta")
        if not os.path.exists(fa):
            failed.append((stem, "no_out"))
            continue
        native = canon[stem]["seq"]
        npos = set(i for i, c in enumerate(native) if c != "N")
        lines = [l.strip() for l in open(fa) if l.strip()]
        seqs = [lines[j + 1] for j, l in enumerate(lines)
                if l.startswith(">id:") and j + 1 < len(lines)
                and len(lines[j + 1]) == len(native)
                and all(c in "ACGU-" for c in lines[j + 1])]
        recs = []
        for s in seqs:
            pos = [i for i in npos if s[i] != '-']
            if not pos:
                continue
            recs.append(sum(1 for i in pos if s[i] == native[i]) / len(pos))
        if not recs:
            failed.append((stem, "no_valid_samples"))
            continue
        results[stem] = {"len": len(native), "recovery_best": max(recs),
                         "recovery_t01_mean": float(np.mean(recs)),
                         "n_samples": len(recs)}
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(stems)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])) if results else None,
        "n_failed": len(failed),
        "note": "AlignIF(重训 epoch79) × RhoFold+ 预测骨架(86_sctm native_*.pdb) 193 链；"
                "canonical native 真值；'N'/'-' 位跳过；n=20 t=0.1 no-MStA",
    }
    out = {"summary": summary, "per_chain": results, "failed": failed}
    tmp = os.path.join(SUR, "100_alignif_predbb.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUR, "100_alignif_predbb.json"))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["links", "run", "metrics"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.phase == "links":
        phase_links()
    elif args.phase == "run":
        phase_run(args.workers)
    elif args.phase == "metrics":
        phase_metrics()
