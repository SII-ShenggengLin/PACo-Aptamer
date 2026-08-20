#!/usr/bin/env python3
"""第 15 步：批量 RhoFold 推理（评估预测器，scTM 用）

输入：survey/eval_inputs/eval_manifest.json + 每条目 JSON（序列）
输出：survey/rhofold_out/{pdb}_{chain}.pdb（unrelaxed，relax_steps=0）

设计：
- 单序列模式（--single_seq_pred True），避免 MSA 数据库依赖
- 逐条链提交（并行 2 进程 × GPU 1/2，用户指定 GPU 编号 1/2）
- 幂等：已存在输出则跳过；失败记录
- 注意：rhofold_service_env 是共享环境（只读使用，不安装任何包）
"""
import os, subprocess, sys, json, time
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT_DIR = os.path.join(DATA_DIR, "survey", "rhofold_out")
os.makedirs(OUT_DIR, exist_ok=True)

RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
RHOFOLD_REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
GPU_IDS = [1, 2]   # 用户指定：编号 1、2 两块 GPU（0、3 被其他任务占用）
MAX_LEN = 1024     # RNA-FM 最大长度

def extract_seqs():
    """从 eval_inputs 提取全部 (pdb, chain, seq)"""
    items = []
    for fn in sorted(os.listdir(os.path.join(DATA_DIR, "survey", "eval_inputs"))):
        if not fn.endswith(".json") or fn == "eval_manifest.json":
            continue
        e = json.load(open(os.path.join(DATA_DIR, "survey", "eval_inputs", fn)))
        for ch, seq in e["sequences"].items():
            items.append((e["pdb"], ch, seq))
    return items

def run_one(item, gpu):
    pdb, ch, seq = item
    out_pdb = os.path.join(OUT_DIR, f"{pdb}_{ch}.pdb")
    if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 1000:
        return (pdb, ch, "cached")
    if len(seq) > MAX_LEN:
        return (pdb, ch, "too_long")
    fa = os.path.join(OUT_DIR, f"{pdb}_{ch}.fa")
    with open(fa, "w") as f:
        f.write(f">{pdb}_{ch}\n{seq}\n")
    workdir = os.path.join(OUT_DIR, f"tmp_{pdb}_{ch}")
    cmd = [RHOFOLD_PY, "inference.py",
           "--input_fas", fa, "--output_dir", workdir,
           "--device", f"cuda:{gpu}", "--single_seq_pred", "True",
           "--relax_steps", "0",
           "--ckpt", f"{RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
    try:
        r = subprocess.run(cmd, cwd=RHOFOLD_REPO, capture_output=True, timeout=600)
        pred = os.path.join(workdir, "unrelaxed_model.pdb")
        if r.returncode != 0 or not os.path.exists(pred):
            return (pdb, ch, f"fail: {r.stderr.decode(errors='ignore')[-200:]}")
        os.rename(pred, out_pdb)
        # 清理中间文件
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        os.rmdir(workdir)
        os.remove(fa)
        return (pdb, ch, "ok")
    except subprocess.TimeoutExpired:
        return (pdb, ch, "timeout")
    except Exception as e:
        return (pdb, ch, f"err: {e}")

def main():
    items = extract_seqs()
    print(f"待推理序列: {len(items)}", flush=True)
    gpu_assign = {i: GPU_IDS[i % len(GPU_IDS)] for i in range(len(items))}
    stats = {"ok": 0, "cached": 0, "too_long": 0, "fail": 0, "timeout": 0}
    with ThreadPoolExecutor(max_workers=2) as ex:   # 2 进程 = GPU 1/2 各一
        futs = {ex.submit(run_one, item, gpu_assign[i]): item for i, item in enumerate(items)}
        for i, fut in enumerate(futs):
            pdb, ch, status = fut.result()
            if status == "ok":
                stats["ok"] += 1
            elif status == "cached":
                stats["cached"] += 1
            elif status == "too_long":
                stats["too_long"] += 1
            elif status == "timeout":
                stats["timeout"] += 1
            else:
                stats["fail"] += 1
                print(f"  [FAIL] {pdb}_{ch}: {status}", flush=True)
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{len(items)} {stats}", flush=True)
    print(f"完成: {stats}", flush=True)
    # 汇总清单
    done = [f[:-4] for f in os.listdir(OUT_DIR) if f.endswith(".pdb")]
    json.dump(done, open(os.path.join(OUT_DIR, "done.json"), "w"))
    print(f"成功预测: {len(done)} 条链", flush=True)

if __name__ == "__main__":
    main()
