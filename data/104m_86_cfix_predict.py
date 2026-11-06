#!/usr/bin/env python3
"""104m：86_sctm 193 链的修正后 gRNAde best 设计 → RhoFold+ 重预测（GPU 2）

每链从 grnade_out_btest_v2_cfix 三温度拼接池（len==native 过滤）选恢复率最高的一条
（86 号 best_design_per_method 的 gRNAde 分支同款口径，native = s1 缓存 seq；
pick_stems 已保证 193 链 s1 与 canonical 序列一致）。
输出：survey/86_sctm/gRNAdeCFIX_{stem}.fa / .pdb（不覆盖任何既有文件；
已有非空产物则跳过，支持断点续跑）。
另写 survey/86_sctm_gfix_designs.json（逐链选中序列 + 恢复率）供审计。

启动：CUDA_VISIBLE_DEVICES=2 python data/104m_86_cfix_predict.py（脚本内 device cuda:0）
"""
import os, sys, json, time, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
SC_DIR = os.path.join(SUR, "86_sctm")
RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
RHOFOLD_REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
TEMPS = [0.1, 0.5, 1.0]
NAME_PREFIX = "gRNAdeCFIX"


def best_design_cfix(stem, native):
    def rec(s):
        return sum(1 for a, b in zip(s, native) if a == b) / len(native)
    pool = []
    for t in TEMPS:
        fp = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.{t}.seqs")
        if os.path.exists(fp):
            pool += [l.strip() for l in open(fp) if l.strip()]
    pool = [s for s in pool if len(s) == len(native)]
    if not pool:
        return None, None
    best = max(pool, key=rec)
    return best, rec(best)


def rhofold_one(name, seq):
    out_pdb = os.path.join(SC_DIR, name + ".pdb")
    if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 1000:
        return "cached"
    if len(seq) > 1024:
        return "too_long"
    fa = os.path.join(SC_DIR, name + ".fa")
    tmp_fa = fa + f".tmp.{os.getpid()}"
    with open(tmp_fa, "w") as f:
        f.write(f">{name}\n{seq}\n")
    os.replace(tmp_fa, fa)
    wd = os.path.join(SC_DIR, "tmp_" + name)
    cmd = [RHOFOLD_PY, "inference.py", "--input_fas", fa, "--output_dir", wd,
           "--device", "cuda:0", "--single_seq_pred", "True", "--relax_steps", "0",
           "--ckpt", f"{RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
    try:
        r = subprocess.run(cmd, cwd=RHOFOLD_REPO, capture_output=True, timeout=600)
        pred = os.path.join(wd, "unrelaxed_model.pdb")
        if r.returncode != 0 or not os.path.exists(pred):
            return "fail"
        shutil.copyfile(pred, out_pdb + f".tmp.{os.getpid()}")
        os.replace(out_pdb + f".tmp.{os.getpid()}", out_pdb)
        shutil.rmtree(wd, ignore_errors=True)
        return "ok"
    except Exception:
        return "fail"


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from PACoAptamer.data_utils import load_s1
    sc86 = json.load(open(os.path.join(SUR, "86_sctm.json")))
    stems = sorted(sc86["per_chain"])
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    print(f"[104m] 193 链子集 n={len(stems)}", flush=True)

    jobs, designs = [], {}
    for stem in stems:
        native = data[stem]["seq"]
        seq, r = best_design_cfix(stem, native)
        if seq is None:
            print(f"  [warn] {stem} 无有效 cfix 池")
            continue
        designs[stem] = {"seq": seq, "recovery_best": r, "len": len(native)}
        jobs.append((f"{NAME_PREFIX}_{stem}", seq))
    tmp = os.path.join(SUR, f"86_sctm_gfix_designs.json.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump({"note": "86_sctm 193 链修正后 gRNAde 池内 best 设计（2026-10-16）",
                   "per_chain": designs}, f, indent=1)
    os.replace(tmp, os.path.join(SUR, "86_sctm_gfix_designs.json"))
    print(f"[104m] 预测任务 {len(jobs)}", flush=True)

    t0 = time.time()
    stats = {}
    with ThreadPoolExecutor(max_workers=6) as ex:   # GPU 2 独占且显存/算力富余；进程瓶颈在模型加载
        futs = [ex.submit(rhofold_one, nm, sq) for nm, sq in jobs]
        for i, f in enumerate(futs):
            r = f.result()
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 20 == 0:
                print(f"[104m] {i+1}/{len(jobs)} {stats} | {time.time()-t0:.0f}s", flush=True)
    print(f"[104m] 完成 {stats} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
