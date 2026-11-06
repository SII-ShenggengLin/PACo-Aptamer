#!/usr/bin/env python3
"""第 84 步：logP+INF 双指标共识筛选演示（蓝图 §五可选加分项 / Fig6e 素材，SynTnpB 式闭环）

PACo ens2 池（77 号，300 设计/链，含逐设计 logP）逐设计回折 INF（对输入 SS）→
四种筛选策略在预算 k 下的池质量（平均恢复率）：
  random（=池均值）/ logP / INF / consensus（logP 与 INF 的秩平均）
oracle（按恢复率取 top-k）作上界。
输出：survey/84_consensus_screen.json（预算曲线 1/5/10/25/50% + 增益与 CI）
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m77 = _imp("m77", os.path.join(ROOT, "data", "77_p0234_analysis.py"))

# 模块级共享：参考 SS
from PACoAptamer.data_utils import load_s1
_SP = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
_DATA = load_s1(set(_SP["test"]), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
_REF_SS = {s: m77.pairs_from_matrix(_DATA[s]["ss_pairs"])
           for s in _DATA if len(_DATA[s]["seq"]) <= 512}


def _worker(stem):
    p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
    if not os.path.exists(p) or stem not in _REF_SS:
        return stem, None
    v = json.load(open(p))
    native, pool, logp = v["native"], v["pool"], v["logp"]
    uniq = sorted({s for s in pool if set(s) <= set("ACGU")})
    if not uniq:
        return stem, None
    structs = m77.fold_batch(uniq)
    if len(structs) != len(uniq):
        return stem, None
    pairmap = {s: m77.pairs_from_dbn(ss) for s, ss in zip(uniq, structs)}
    good_idx = [i for i, s in enumerate(pool) if s in pairmap]
    pool_g = [pool[i] for i in good_idx]
    rec = np.array([sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool_g])
    inf = np.array([m77.inf_scores(pairmap[s], _REF_SS[stem])["inf"] for s in pool_g])
    lp = np.array([logp[i] for i in good_idx])
    if np.std(lp) == 0 or len(pool) < 50:
        return stem, None
    # 秩平均共识
    rl = np.argsort(np.argsort(-lp))
    ri = np.argsort(np.argsort(-inf))
    cons = rl + ri
    return stem, {"rec": rec, "inf": inf, "lp": lp, "cons": cons}


def main():
    # load_s1 命中缓存返回全量 6,338 链(忽略 subset) → 必须与 test 交集
    # （2026-08-19 修复：曾直接 sorted(_REF_SS)，靠池文件存在性侥幸兜底）
    stems = sorted(set(_SP["test"]) & set(_REF_SS))
    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, r) in enumerate(ex.map(_worker, stems, chunksize=4)):
            if r:
                per_chain[stem] = r
            if (i + 1) % 100 == 0:
                print(f"[84] {i+1}/{len(stems)} | {time.time()-t0:.0f}s", flush=True)
    budgets = [0.01, 0.05, 0.10, 0.25, 0.50]
    out = {"n_chains": len(per_chain), "curves": {}}
    for strat, key in (("random", None), ("logP", "lp"), ("INF", "inf"),
                       ("consensus", "cons"), ("oracle", "rec")):
        cur = {}
        for b in budgets:
            gains = []
            for stem, d in per_chain.items():
                n = len(d["rec"])
                k = max(1, int(round(n * b)))
                if strat == "random":
                    sel_mean = d["rec"].mean()
                else:
                    order = np.argsort(-d[key])[:k]
                    sel_mean = d["rec"][order].mean()
                gains.append(sel_mean - d["rec"].mean())
            rng = np.random.RandomState(0)
            bs = [rng.choice(gains, len(gains)).mean() for _ in range(2000)]
            cur[f"budget_{b:.2f}"] = {"gain": float(np.mean(gains)),
                                      "ci95": [float(np.percentile(bs, 2.5)),
                                               float(np.percentile(bs, 97.5))]}
        out["curves"][strat] = cur
    json.dump(out, open(os.path.join(SUR, "84_consensus_screen.json"), "w"), indent=1)
    for strat in ("random", "logP", "INF", "consensus", "oracle"):
        c = out["curves"][strat]
        line = " ".join(f"{b:.0%}:{c[f'budget_{b:.2f}']['gain']:+.4f}" for b in [0.01, 0.05, 0.1, 0.25, 0.5])
        print(f"[84] {strat:<10} {line}")


if __name__ == "__main__":
    main()
