#!/usr/bin/env python3
"""第 97 步：PACo(ens2) 对 SS 输入的稳健性（test 477 链 × 4 条件 × 300 样本）

条件（特征 = load_s1 缓存，唯一变量 = ss_pairs）：
  standard  缓存原 SS（RNAfold(native) MFE）——参照，应复现主表 74.90/60.40
  zero      全零配对矩阵（无 SS 信息）
  shuffle   位置索引随机排列置换配对矩阵行列（保持配对总数、破坏模式；
            torch.Generator 固定种子 9701，与采样流独立）
  centroid  RNAfold --noPS -p2 的 centroid 结构（花括号行）替代 MFE 作 SS

采样：ens2 主配置（seed43_noiface + ssdo15_noiface 逐步概率平均），
torch.manual_seed(42) 全局一次，链顺序 = list(splits_v2 test) 过滤（与 77 号
phase_pool 完全一致 → 条件 1 应逐链级复现 77 号池），3 温度×100。

输出：survey/97_ss_robust.json（protocol/validation/summary/per_chain）
用法：CUDA_VISIBLE_DEVICES=0 python data/97_ss_robust.py [--subset N]
"""
import os, sys, json, time, argparse, subprocess, re
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100
MAX_LEN = 512
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
ENS_CKPTS = ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]
CONDS = ["standard", "zero", "shuffle", "centroid"]

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m39 = _imp("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
from PACoAptamer.data_utils import load_s1, ss_to_pairs


def centroid_ss(seq):
    """RNAfold --noPS -p2 → centroid 结构；失败回退 MFE，再回退全点。
    坑（1QA6_C 实录）：centroid 行判定必须看 ` d=` 后缀（形如 `{ -7.40 d=3.88}`），
    否则 PF 行（结构本身可能含 {} 配对字符）会被 `\{` 回溯误匹配 → 长度错 → GVP 炸。
    另加长度校验双保险。"""
    L = len(seq)
    try:
        r = subprocess.run([RNAFOLD, "--noPS", "-p2"], input=seq + "\n",
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return "." * L
    lines = r.stdout.splitlines()
    mfe = None
    for l in lines[1:]:
        if mfe is None:
            m = re.match(r"^([\.\(\)\[\]\{\}\<\>]+)\s*\(", l)
            if m and len(m.group(1)) == L:
                mfe = m.group(1)
        mc = re.match(r"^([\.\(\)\[\]\{\}\<\>]+)\s+\{[^{}]*\bd=", l)
        if mc and len(mc.group(1)) == L:
            return mc.group(1)
    return mfe if mfe else "." * L


def boot_ci(x, n=2000, seed=0):
    if len(x) < 3:
        return None
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="只跑前 N 链（调试）")
    ap.add_argument("--conds", type=str, default=",".join(CONDS))
    args = ap.parse_args()
    conds = args.conds.split(",")

    device = 0
    torch.manual_seed(42)
    models = []
    alphabet = None
    for n in ENS_CKPTS:
        p = os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m74.load_model(p, device)
        models.append(m)

    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test_stems = list(sp2["test"])
    data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    test_stems = [s for s in test_stems if s in data]   # 与 77 号同顺序同过滤
    if args.subset:
        test_stems = test_stems[:args.subset]
    print(f"[97] 链 {len(test_stems)}，条件 {conds}", flush=True)

    chain_dir = os.path.join(SUR, "97_ss_robust_chains")
    os.makedirs(chain_dir, exist_ok=True)
    per_chain = {}
    for s in test_stems:
        p = os.path.join(chain_dir, s + ".json")
        if os.path.exists(p):
            per_chain[s] = json.load(open(p))
    print(f"[97] 断点命中 {len(per_chain)} 链", flush=True)
    t0 = time.time()
    n_new = 0
    for i, stem in enumerate(test_stems):
        if stem in per_chain and all(per_chain[stem].get(c) for c in conds):
            continue
        d = data[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        native = d['seq']
        L = len(native)
        rec = {"len": L}
        ss_variants = {}
        for cond in conds:
            if cond == "standard":
                ss = d['ss_pairs']
            elif cond == "zero":
                ss = np.zeros_like(d['ss_pairs'])
            elif cond == "shuffle":
                g = torch.Generator().manual_seed(9701)
                perm = torch.randperm(L, generator=g).numpy()
                ss = np.asarray(d['ss_pairs'])[perm][:, perm]
            elif cond == "centroid":
                if stem not in ss_variants:
                    ss_variants[stem] = ss_to_pairs(centroid_ss(native))
                ss = ss_variants[stem]
            d2 = {"coords9": d['coords9'], "ss_pairs": ss, "seq": native}
            try:
                pool = []
                for t in TEMPS:
                    pool.extend(m39.sample_chain_ens(models, alphabet, d2, device, t, N_PER_TEMP))
            except RuntimeError as e:
                print(f"  [跳过 {stem}/{cond}: {e}]", flush=True)
                rec[cond] = None
                continue
            recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
            rec[cond] = {"best": float(max(recs)), "t01": float(np.mean(recs[:N_PER_TEMP]))}
        per_chain[stem] = rec
        tmpc = os.path.join(chain_dir, stem + ".json.tmp")
        json.dump(rec, open(tmpc, "w"))
        os.replace(tmpc, os.path.join(chain_dir, stem + ".json"))
        n_new += 1
        if n_new % 50 == 0:
            import torch as _t
            _t.cuda.empty_cache()   # 防 caching allocator 碎片膨胀
        if (i + 1) % 25 == 0:
            print(f"[97] {i+1}/{len(test_stems)} | {time.time()-t0:.0f}s", flush=True)

    # ---- 汇总 ----
    summary = {}
    for cond in conds:
        bs = [r[cond]["best"] for r in per_chain.values() if r.get(cond)]
        ts = [r[cond]["t01"] for r in per_chain.values() if r.get(cond)]
        row = {"n": len(bs), "best_mean": float(np.mean(bs)), "t01_mean": float(np.mean(ts)),
               "best_ci95": boot_ci(bs), "t01_ci95": boot_ci(ts)}
        if cond != "standard":
            db = [r[cond]["best"] - r["standard"]["best"] for r in per_chain.values()
                  if r.get(cond) and r.get("standard")]
            dt = [r[cond]["t01"] - r["standard"]["t01"] for r in per_chain.values()
                  if r.get(cond) and r.get("standard")]
            row["delta_best_vs_standard"] = {"mean": float(np.mean(db)), "ci95": boot_ci(db),
                                             "median": float(np.median(db))}
            row["delta_t01_vs_standard"] = {"mean": float(np.mean(dt)), "ci95": boot_ci(dt),
                                            "median": float(np.median(dt))}
        summary[cond] = row

    # ---- 复现校验：standard 条件 vs 77 号池 + 主表 ----
    val = {}
    pool77_dir = os.path.join(SUR, "77_ens2_pool_btest")
    n_match77, n_cmp77 = 0, 0
    diffs_b = []
    for stem, r in per_chain.items():
        if not r.get("standard"):
            continue
        p = os.path.join(pool77_dir, stem + ".json")
        if os.path.exists(p):
            v77 = json.load(open(p))
            n_cmp77 += 1
            if abs(v77["recovery_best"] - r["standard"]["best"]) < 1e-9 and \
               abs(v77["recovery_t01_mean"] - r["standard"]["t01"]) < 1e-9:
                n_match77 += 1
            diffs_b.append(abs(v77["recovery_best"] - r["standard"]["best"]))
    val["vs_77pool"] = {"n_compared": n_cmp77, "n_exact_match": n_match77,
                        "mean_abs_diff_best": float(np.mean(diffs_b)) if diffs_b else None}
    if summary.get("standard"):
        val["standard_vs_maintab"] = {
            "best": summary["standard"]["best_mean"], "t01": summary["standard"]["t01_mean"],
            "maintab_best": 0.7490, "maintab_t01": 0.6040}

    out = {
        "protocol": {"models": ENS_CKPTS, "temps": TEMPS, "n_per_temp": N_PER_TEMP,
                     "seed": 42, "shuffle_seed": 9701,
                     "note": "特征=load_s1 缓存；唯一变量=ss_pairs；链顺序同 77 号"},
        "validation": val,
        "summary": summary,
        "per_chain": per_chain,
    }
    tmp = os.path.join(SUR, "97_ss_robust.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUR, "97_ss_robust.json"))
    print(json.dumps({"validation": val, "summary": summary}, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
