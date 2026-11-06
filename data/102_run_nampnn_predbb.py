#!/usr/bin/env python3
"""第 102 步：NA-MPNN 在 RhoFold+ 预测骨架上的批量推理（86_sctm 193 链）

- 输入准备：survey/102_predbb_inputs/{stem}.pdb 符号链接 → survey/86_sctm/native_{stem}.pdb
- 推理：仿 59 号（逐链子进程 design 模式，t=0.1 × 10 样本，单链 timeout 900s，workers 4）
- 指标：与 btest_v2_main_table per_chain 同口径（'N' 位跳过——193 链无 N；
  best=池内最大同一性，t01 mean=池逐样本平均；native=canonical_v2.json）
  全长样本优先；无全长样本时回退 61 号"有效位口径"（parse_PDB 有效位对齐）
- 效率探针：前 6 条链墙钟 + GPU 显存峰值（轮询 nvidia-smi）→ survey/99_efficiency_NA-MPNN.json
输出：survey/nampnn_out_predbb/{stem}.seqs + survey/102_nampnn_predbb.json
用法：CUDA_VISIBLE_DEVICES=2 python data/102_run_nampnn_predbb.py --gpu 2 [--workers 4] [--skip-inference]
"""
import os, sys, json, subprocess, shutil, argparse, time, threading, difflib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
NAMPNN_DIR = os.path.join(ROOT, "tools", "NA-MPNN")
SRC_BB = os.path.join(SUR, "86_sctm")
IN_DIR = os.path.join(SUR, "102_predbb_inputs")
OUT_DIR = os.path.join(SUR, "nampnn_out_predbb")
CKPT = os.path.join(ROOT, "models", "NA-MPNN_btrain_v2", "s_11016.pt")
# aptamer_eval 环境已删除：odesign_service_env + pylibs(prody)
PY = "/root/miniconda3/envs/odesign_service_env/bin/python"
PYLIBS = os.path.join(SUR, "pylibs")

RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}

restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
            'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
            'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
       'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
alphabet = [r31[r] for r in restypes]


def load_chains():
    d = json.load(open(os.path.join(SUR, "86_sctm.json")))
    return sorted(d["per_chain"].keys())


def prep_inputs(chains):
    """符号链接 86_sctm/native_{stem}.pdb → 102_predbb_inputs/{stem}.pdb"""
    os.makedirs(IN_DIR, exist_ok=True)
    n_new = 0
    for stem in chains:
        src = os.path.join(SRC_BB, f"native_{stem}.pdb")
        dst = os.path.join(IN_DIR, f"{stem}.pdb")
        if not os.path.exists(src):
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
        n_new += 1
    print(f"[prep] 输入链接 {len(chains)} 链（新建 {n_new}）", flush=True)


def parse_fa(fa_path):
    """59 号口径：id= 行下一行为样本，单字母码还原 A/C/G/U"""
    lines = [l.strip() for l in open(fa_path) if l.strip()]
    seqs = []
    for j, line in enumerate(lines):
        if line.startswith(">") and "id=" in line and j + 1 < len(lines):
            seq = "".join(RNA_MAP.get(c, "") for c in lines[j + 1])
            if len(seq) > 0:
                seqs.append(seq)
    return seqs


def run_one(stem, gpu, probe_rec=None):
    out_f = os.path.join(OUT_DIR, f"{stem}.seqs")
    if os.path.exists(out_f) and os.path.getsize(out_f) > 100:
        return (stem, "cached")
    pdb_path = os.path.join(IN_DIR, f"{stem}.pdb")
    if not os.path.exists(pdb_path):
        return (stem, "no_pdb")
    n_atom = sum(1 for l in open(pdb_path) if l.startswith("ATOM"))
    if n_atom == 0:
        return (stem, "no_coords")
    workdir = os.path.join(OUT_DIR, f"tmp_{stem}")
    cmd = [PY, "./inference/run.py",
           "--model_type", "na_mpnn",
           "--mode", "design",
           "--checkpoint_na_mpnn", CKPT,
           "--batch_size", "10",
           "--pdb_path", pdb_path,
           "--out_folder", workdir,
           "--output_pdbs", "0"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": PYLIBS}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=NAMPNN_DIR, capture_output=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return (stem, "timeout")
    wall = time.time() - t0
    fa = os.path.join(workdir, "seqs", f"{stem}.fa")
    if r.returncode != 0 or not os.path.exists(fa):
        print(f"    [诊断] {stem} rc={r.returncode} stderr={r.stderr.decode()[-300:]!r}", flush=True)
        shutil.rmtree(workdir, ignore_errors=True)
        return (stem, "failed")
    seqs = parse_fa(fa)
    shutil.rmtree(workdir, ignore_errors=True)
    if not seqs:
        return (stem, "no_output")
    tmp_f = out_f + ".tmp"
    with open(tmp_f, "w") as f:
        f.write("\n".join(seqs) + "\n")
    os.rename(tmp_f, out_f)
    if probe_rec is not None:
        probe_rec["wall_s"] = wall
    return (stem, f"ok_{len(seqs)}")


# ---------- 效率探针：轮询 nvidia-smi 取显存峰值 ----------

def gpu_mem_used_mib(gpu):
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                        "-i", str(gpu)], capture_output=True, text=True)
    return float(r.stdout.strip())


def probe(chains, gpu, n=6):
    recs = []
    for stem in chains[:n]:
        pr = {}
        peak = [0.0]
        stop = threading.Event()
        base = gpu_mem_used_mib(gpu)

        def poll():
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], gpu_mem_used_mib(gpu) - base)
                except Exception:
                    pass
                stop.wait(0.2)

        th = threading.Thread(target=poll)
        th.start()
        _, status = run_one(stem, gpu, probe_rec=pr)
        stop.set()
        th.join()
        canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
        recs.append({"stem": stem, "status": status, "len": len(canon[stem]["seq"]),
                     "wall_s": round(pr.get("wall_s", -1), 3),
                     "gpu_mem_peak_mib_polling": round(peak[0], 1)})
        print(f"  [probe] {stem}: {status} wall={pr.get('wall_s', -1):.2f}s mem~{peak[0]:.0f}MiB", flush=True)
    out = {"method": "NA-MPNN", "setting": "predbb t=0.1 x10 batch_size=10",
           "gpu": gpu, "note": "显存为 nvidia-smi 0.2s 轮询差值，近似峰值",
           "records": recs}
    fp = os.path.join(SUR, "99_efficiency_NA-MPNN.json")
    tmp = fp + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.rename(tmp, fp)
    print(f"[probe] 写入 {fp}", flush=True)


# ---------- 指标（61/63 号口径） ----------

def parse_native_predbb(stem):
    """无全长样本链回退：NA-MPNN parse_PDB 有效位（61 号口径，作用于预测骨架 PDB）"""
    cwd = os.getcwd()
    os.chdir(os.path.join(NAMPNN_DIR, "inference"))
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, PYLIBS)
    from data_utils import parse_PDB
    d, _, _, _, _ = parse_PDB(os.path.join(IN_DIR, f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    os.chdir(cwd)
    S = d["S"].cpu().numpy()
    return "".join(RNA_MAP.get(alphabet[AA], "?") for AA in S)


def align_positions(parsed, native):
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos, ppos = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
            ppos.extend(range(j1, j2))
    return pos


def compute_metrics(chains, gpu):
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    results, gap_chains, failed = {}, {}, []
    for stem in chains:
        out_f = os.path.join(OUT_DIR, f"{stem}.seqs")
        if not os.path.exists(out_f):
            failed.append(stem)
            continue
        native = canon[stem]["seq"]
        npos = [i for i, c in enumerate(native) if c != "N"]
        seqs = [l.strip() for l in open(out_f) if l.strip()]
        full = [s for s in seqs if len(s) == len(native)]
        if full:
            recs = [sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in full]
        else:
            try:
                parsed = parse_native_predbb(stem)
            except Exception as e:
                print(f"  [指标] {stem} parse 失败: {e}", flush=True)
                failed.append(stem)
                continue
            pos_all = align_positions(parsed, native)
            sub = [s for s in seqs if len(s) == len(pos_all)]
            if not sub or not pos_all:
                failed.append(stem)
                continue
            pos = [i for i in pos_all if native[i] != "N"]
            idx = {i: k for k, i in enumerate(pos_all)}
            recs = [sum(1 for i in pos if s[idx[i]] == native[i]) / len(pos) for s in sub]
            gap_chains[stem] = len(pos)
        results[stem] = {"len": len(native), "recovery_best": max(recs),
                         "recovery_t01_mean": float(np.mean(recs)), "n_samples": len(recs)}
    summary = {
        "n_chains": len(results), "coverage": f"{len(results)}/{len(chains)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])) if results else None,
        "note": "RhoFold+ 预测骨架(86_sctm) × NA-MPNN t=0.1x10；native=canonical_v2；'N' 位跳过；口径同 btest_v2_main_table",
        "failed_chains": failed, "gap_chains": gap_chains,
    }
    fp = os.path.join(SUR, "102_nampnn_predbb.json")
    tmp = fp + ".tmp"
    json.dump({"summary": summary, "per_chain": results}, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.rename(tmp, fp)
    print(f"[指标] 写入 {fp}: best_mean={summary['recovery_best_mean']}, t01_mean={summary['recovery_t01_mean_mean']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()
    chains = load_chains()
    os.makedirs(OUT_DIR, exist_ok=True)
    prep_inputs(chains)
    if not args.skip_probe:
        probe(chains, args.gpu)
    if not args.skip_inference:
        stats = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, s, args.gpu): s for s in chains}
            for i, fut in enumerate(futs):
                stem, status = fut.result()
                k = status.split("_")[0]
                stats[k] = stats.get(k, 0) + 1
                if (i + 1) % 25 == 0:
                    print(f"进度 {i+1}/{len(chains)} {stats}", flush=True)
        print(f"[NAMPNN-predbb GPU{args.gpu}] 完成: {stats}", flush=True)
    compute_metrics(chains, args.gpu)
