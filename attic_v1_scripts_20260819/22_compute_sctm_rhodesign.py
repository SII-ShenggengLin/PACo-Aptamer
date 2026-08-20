#!/usr/bin/env python3
"""第 19 步：scTM 指标（设计序列 → RhoFold 预测 → US-align vs 天然结构）

- gRNAde 设计序列（每链取 best 恢复率样本）→ RhoFold 预测（relax_steps=0, GPU1/2）
- US-align vs 天然链 PDB → TM score
- 同时计算 ground truth 自洽上界（天然序列的 RhoFold 预测 vs 天然结构，15 号产物）

输出：survey/metrics_RhoDesign_scTM.json
"""
import os, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
GR_OUT = os.path.join(DATA_DIR, "survey", "rhodesign_out")
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs")
RF_OUT = os.path.join(DATA_DIR, "survey", "rhofold_out")
SC_OUT = os.path.join(DATA_DIR, "survey", "sctm_out")
os.makedirs(SC_OUT, exist_ok=True)

USALIGN = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/USalign/USalign"
RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
RHOFOLD_REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
GPU_IDS = [1, 2]
TEMPS = [0.1, 0.5, 1.0]

def best_seq(stem):
    """池内 best 恢复率样本"""
    native_p = os.path.join(GR_OUT, f"{stem}.native")
    if not os.path.exists(native_p):
        return None, None
    native = open(native_p).read().strip()
    best, best_r = None, -1
    for t in TEMPS:
        sp = os.path.join(GR_OUT, f"{stem}.{t}.seqs")
        if not os.path.exists(sp):
            continue
        for s in (l.strip() for l in open(sp) if l.strip()):
            r = sum(1 for a, b in zip(s, native) if a == b) / max(1, len(native))
            if r > best_r:
                best, best_r = s, r
    return best, native

def usalign_tm(pdb1, pdb2):
    r = subprocess.run([USALIGN, pdb1, pdb2, "-outfmt", "2"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    for line in r.stdout.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            return float(parts[2])  # TM1
    return None

def rhofold_predict(seq, stem, gpu):
    """RhoFold 预测单序列 → PDB 路径"""
    fa = os.path.join(SC_OUT, f"{stem}.fa")
    with open(fa, "w") as f:
        f.write(f">{stem}\n{seq}\n")
    workdir = os.path.join(SC_OUT, f"tmp_{stem}")
    cmd = [RHOFOLD_PY, "inference.py",
           "--input_fas", fa, "--output_dir", workdir,
           "--device", f"cuda:{gpu}", "--single_seq_pred", "True",
           "--relax_steps", "0",
           "--ckpt", f"{RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
    pred = os.path.join(workdir, "unrelaxed_model.pdb")
    r = subprocess.run(cmd, cwd=RHOFOLD_REPO, capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(pred):
        return None
    out = os.path.join(SC_OUT, f"{stem}_pred.pdb")
    os.rename(pred, out)
    for f in os.listdir(workdir):
        os.remove(os.path.join(workdir, f))
    os.rmdir(workdir)
    os.remove(fa)
    return out

def work_one(stem, gpu):
    result = {"stem": stem}
    # ground truth 自洽上界（天然序列的 RhoFold 预测）
    nat_pred = os.path.join(RF_OUT, f"{stem}.pdb")
    nat_pdb = os.path.join(IN_DIR, f"{stem}.pdb")
    if os.path.exists(nat_pred) and os.path.exists(nat_pdb):
        result["sctm_self_consistency"] = usalign_tm(nat_pdb, nat_pred)
    # 设计序列 scTM
    best, native = best_seq(stem)
    if best is None:
        return (stem, result)
    pred_pdb = os.path.join(SC_OUT, f"{stem}_pred.pdb")
    if not os.path.exists(pred_pdb):
        pred_pdb = rhofold_predict(best, stem, gpu)
        if pred_pdb is None:
            result["sctm_design"] = None
            return (stem, result)
    if os.path.exists(nat_pdb):
        result["sctm_design"] = usalign_tm(nat_pdb, pred_pdb)
    return (stem, result)

def main():
    stems = sorted(f[:-7] for f in os.listdir(GR_OUT) if f.endswith(".native"))
    # 只做已有 gRNAde 输出的链
    stems = [s for s in stems if any(os.path.exists(os.path.join(GR_OUT, f"{s}.{t}.seqs")) for t in TEMPS)]
    print(f"scTM 计算 {len(stems)} 条链", flush=True)
    results = {}
    gpu_assign = {i: GPU_IDS[i % 2] for i in range(len(stems))}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(work_one, stem, gpu_assign[i]): stem for i, stem in enumerate(stems)}
        for i, fut in enumerate(futs):
            stem, r = fut.result()
            results[stem] = r
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{len(stems)}", flush=True)
    # 汇总
    sc = [r["sctm_design"] for r in results.values() if r.get("sctm_design")]
    ssc = [r["sctm_self_consistency"] for r in results.values() if r.get("sctm_self_consistency")]
    summary = {
        "n": len(results),
        "sctm_design_mean": sum(sc) / len(sc) if sc else None,
        "sctm_design_pass_045": sum(1 for v in sc if v >= 0.45) / len(sc) if sc else None,
        "sctm_self_consistency_mean": sum(ssc) / len(ssc) if ssc else None,
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_RhoDesign_scTM.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== RhoDesign scTM =====", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)

if __name__ == "__main__":
    main()
