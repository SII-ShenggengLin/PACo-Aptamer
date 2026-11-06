#!/usr/bin/env python3
"""第 86 步：Fig4f 自洽结构评估 + Fig4g recovery-matched 随机负对照（v2 口径）

B 层 test 477 分层抽样 200 链 × 6 条件：
  native            天然序列（canonical 序，与 canonical_v2 PDB 对齐）→ 自洽上界
  PACo / RhoDesign / gRNAde / NA-MPNN   各方法池内 best 恢复率设计
  randmatch         PACo best 设计的 recovery-matched 随机对照：
                    保留匹配位，非匹配位随机换成"错误"碱基（同恢复率、序列内容随机）

流程：RhoFold+ 单序列预测（GPU 1/2 并行）→ USalign TM vs canonical_v2 天然 PDB。
输出：survey/86_sctm/{cond}_{stem}.pdb + survey/86_sctm.json（逐链 TM + 汇总）

环境：/root/miniconda3/envs/aptamer_eval/bin/python；GPU 1/2（脚本内部 device cuda:0/1，
须 CUDA_VISIBLE_DEVICES=1,2 启动）
"""
import os, sys, json, time, argparse, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
SC_DIR = os.path.join(SUR, "86_sctm")
CAN_DIR = os.path.join(DATA_DIR, "canonical_v2")
USALIGN = os.path.join(ROOT, "tools", "USalign", "USalign")
RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
RHOFOLD_REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
N_SUB = 200
TEMPS = [0.1, 0.5, 1.0]

sys.path.insert(0, DATA_DIR)
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m77 = _imp("m77", os.path.join(DATA_DIR, "77_p0234_analysis.py"))


def pick_stems():
    """分层抽样 200：按长度四分层 × 界面/非界面，canonical 序与缓存序一致的链"""
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = list(sp["test"])
    can = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    iface_meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    iface = {k for k, v in iface_meta.items() if v and v.get("usable") and 0 < v["n_iface"] <= 50}
    from PACoAptamer.data_utils import load_s1
    data = load_s1(set(test), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    cand = [s for s in test if s in data and s in can
            and can[s]["seq"] == data[s]["seq"]          # 两套口径一致（避免上界被错配拉低）
            and 20 <= len(data[s]["seq"]) <= 300]
    rng = np.random.RandomState(42)
    lens = np.array([len(data[s]["seq"]) for s in cand])
    q = np.quantile(lens, [0.25, 0.5, 0.75])
    strat = [(min(int(np.searchsorted(q, L)), 3), s in iface) for s, L in zip(cand, lens)]
    order = rng.permutation(len(cand))          # 索引置换（避免 numpy shuffle 展开元组）
    picked, per = [], {}
    for i in order:
        st, s = strat[i], cand[i]
        per.setdefault(st, 0)
        if per[st] < N_SUB // 8 + 2:   # 8 个格子大致均衡
            picked.append(s)
            per[st] += 1
        if len(picked) >= N_SUB:
            break
    return picked


def best_design_per_method(stem, native):
    """各方法池内 best 恢复率设计 + recovery-matched 随机对照"""
    def rec(s):
        return sum(1 for a, b in zip(s, native) if a == b) / len(native)
    out = {}
    # PACo ens2 池
    v = json.load(open(os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")))
    tbest = max(v["pool"], key=rec)
    out["PACo"] = tbest
    # recovery-matched 随机：匹配位保留，非匹配位随机换成错误碱基
    rng = np.random.RandomState(zlib_crc(stem))
    rm = []
    for i, (a, b) in enumerate(zip(tbest, native)):
        if a == b:
            rm.append(b)
        else:
            wrong = [c for c in "ACGU" if c != b]
            rm.append(rng.choice(wrong))
    out["randmatch"] = "".join(rm)
    for m, fn in (("RhoDesign", lambda s: [l.strip() for t in TEMPS for l in open(f"{SUR}/rhodesign_out_btest_v2/{s}.{t}.seqs")]),
                  ("gRNAde",    lambda s: [l.strip() for t in TEMPS for l in open(f"{SUR}/grnade_out_btest_v2/{s}.{t}.seqs")]),
                  ("NA-MPNN",   lambda s: [l.strip() for l in open(f"{SUR}/nampnn_out_btest_v2/{s}.seqs")])):
        try:
            pool = [s for s in fn(stem) if len(s) == len(native)]
            out[m] = max(pool, key=rec) if pool else None
        except FileNotFoundError:
            out[m] = None
    return out, rec(tbest)


def zlib_crc(s):
    import zlib
    return zlib.crc32(s.encode()) % (2**31)


def rhofold_one(name, seq, gpu):
    out_pdb = os.path.join(SC_DIR, name + ".pdb")
    if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 1000:
        return "cached"
    if len(seq) > 1024:
        return "too_long"
    fa = os.path.join(SC_DIR, name + ".fa")
    with open(fa, "w") as f:
        f.write(f">{name}\n{seq}\n")
    wd = os.path.join(SC_DIR, "tmp_" + name)
    cmd = [RHOFOLD_PY, "inference.py", "--input_fas", fa, "--output_dir", wd,
           "--device", f"cuda:{gpu}", "--single_seq_pred", "True", "--relax_steps", "0",
           "--ckpt", f"{RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
    try:
        r = subprocess.run(cmd, cwd=RHOFOLD_REPO, capture_output=True, timeout=600)
        pred = os.path.join(wd, "unrelaxed_model.pdb")
        if r.returncode != 0 or not os.path.exists(pred):
            return "fail"
        os.rename(pred, out_pdb)
        shutil.rmtree(wd, ignore_errors=True)
        return "ok"
    except Exception:
        return "fail"


def usalign_tm(native_pdb, pred_pdb):
    """19 号同款：native 在前（TM1 = 按天然链长度归一），-outfmt 2 解析第 3 列"""
    r = subprocess.run([USALIGN, native_pdb, pred_pdb, "-outfmt", "2"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    for line in r.stdout.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            return float(parts[2])
    return None


def _usalign_tm_worker(a):
    """模块级（pickle）：native 在前，TM1"""
    name, pp, np_, stem, cond = a
    try:
        return name, stem, cond, usalign_tm(np_, pp)
    except Exception:
        return name, stem, cond, None


def main():
    os.makedirs(SC_DIR, exist_ok=True)
    stems = pick_stems()
    print(f"[86] 抽样 {len(stems)} 链", flush=True)
    from PACoAptamer.data_utils import load_s1
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp["test"]), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    can = json.load(open(os.path.join(SUR, "canonical_v2.json")))

    jobs = []   # (name, seq)
    meta = {}
    for stem in stems:
        native = data[stem]["seq"]
        designs, tbest_rec = best_design_per_method(stem, native)
        jobs.append((f"native_{stem}", can[stem]["seq"]))
        meta.setdefault(stem, {})["paco_best_recovery"] = tbest_rec
        for m, seq in designs.items():
            if seq:
                jobs.append((f"{m}_{stem}", seq))
    print(f"[86] 预测任务 {len(jobs)}", flush=True)

    t0 = time.time()
    stats = {"ok": 0, "cached": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=4) as ex:   # 每卡 2 并发（外部任务抢资源，提并发保吞吐）
        futs = [ex.submit(rhofold_one, nm, sq, i % 2) for i, (nm, sq) in enumerate(jobs)]
        for i, f in enumerate(futs):
            stats[f.result()] = stats.get(f.result(), 0) + 1
            if (i + 1) % 100 == 0:
                print(f"[86-fold] {i+1}/{len(jobs)} {stats} | {time.time()-t0:.0f}s", flush=True)
    print(f"[86-fold] 完成 {stats} | {time.time()-t0:.0f}s", flush=True)

    # USalign 逐链（CPU 并行 12）
    from concurrent.futures import ProcessPoolExecutor
    args = [(f"{cond}_{stem}", os.path.join(SC_DIR, f"{cond}_{stem}.pdb"),
             os.path.join(CAN_DIR, stem + ".pdb"), stem, cond)
            for stem in stems for cond in ("native", "PACo", "RhoDesign", "gRNAde",
                                           "NA-MPNN", "randmatch")
            if os.path.exists(os.path.join(SC_DIR, f"{cond}_{stem}.pdb"))]

    per_chain = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for name, stem, cond, tm in ex.map(_usalign_tm_worker, args, chunksize=8):
            if tm is not None:
                per_chain.setdefault(stem, {})[f"sctm_{cond}"] = tm
    # 汇总
    out = {"n_stems": len(stems), "per_chain": per_chain, "summary": {}}
    for cond in ("native", "PACo", "RhoDesign", "gRNAde", "NA-MPNN", "randmatch"):
        vals = [v[f"sctm_{cond}"] for v in per_chain.values() if f"sctm_{cond}" in v]
        if vals:
            out["summary"][cond] = {"n": len(vals), "mean": float(np.mean(vals)),
                                    "median": float(np.median(vals)),
                                    "pass_045": float(np.mean(np.array(vals) >= 0.45)),
                                    "pass_031": float(np.mean(np.array(vals) >= 0.31))}
    # Fig4g 配对：PACo − randmatch
    from scipy.stats import wilcoxon
    dif = [v["sctm_PACo"] - v["sctm_randmatch"] for v in per_chain.values()
           if "sctm_PACo" in v and "sctm_randmatch" in v]
    if len(dif) >= 10:
        out["paco_minus_randmatch"] = {
            "n": len(dif), "mean": float(np.mean(dif)),
            "ci95": list(np.percentile(np.random.RandomState(0).choice(dif, (2000, len(dif))).mean(axis=1), [2.5, 97.5])),
            "wilcoxon_p_one_sided": float(wilcoxon(dif, alternative="greater").pvalue)}
    json.dump(out, open(os.path.join(SUR, "86_sctm.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    if "paco_minus_randmatch" in out:
        print("PACo−randmatch:", out["paco_minus_randmatch"])


if __name__ == "__main__":
    main()
