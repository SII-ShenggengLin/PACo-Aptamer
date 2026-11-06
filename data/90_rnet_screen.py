#!/usr/bin/env python3
"""第 90 步:RNet 化学映射信号加入设计筛选面板(Fig6e,全策略同口径重算)

⚠️ 2026-08-31 最终状态:已否决,未运行。89 号泄露审计 verdict=FAIL——
RMDB/Eterna 化学映射库(RNet 训练数据上游)以高相似/嵌入形式包含 56%
测试层原生序列(test 73% / zs 68% / indep 21%),存在记忆泄露通道
(用户决策:有泄露即不加)。脚本保留作设计记录,入口 assert 拦截重跑;
详见 analysis/20 号报告。

前提:89_rnet_leakage_audit.py 审计 PASS(RMDB 全库实测构建序列与三测试层
原生序列零精确/高相似/嵌入重叠——RNet 训练数据不含本项目测试原生序列)。

新信号(OpenKnot 式, 无阈值版):RibonanzaNet 预测设计序列的 2A3(SHAPE 类)
逐核苷酸反应性 → 对输入 SS(条件口径,与 INF 同一参考)配对状态的 AUROC:
反应性高 ⇒ 应未配对。信号只依赖设计序列与参考 SS,不含任何原生序列信息。

为与 84 号曲线可比,本脚本对 PACo ens2-noiface 池(77 号,test 477 链 × 300
设计)在同一设计子集上重算全部策略:
  random / logP / INF / RNET / consensus2(logP+INF 秩平均, = 84 号口径)
  / consensus3(logP+INF+RNET 秩平均) / oracle
预算 1/5/10/25/50% 增益 + bootstrap CI(84 号同款 schema)。

输出:survey/90_rnet_screen.json
运行:GPU(仅 1/2 号卡),后台 nohup,全程约 10-20 分钟。
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "tools", "RibonanzaNet"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "data"))

from types import SimpleNamespace
from Network import RibonanzaNet
from scipy.stats import rankdata, spearmanr
from PACoAptamer.data_utils import load_s1
import importlib.util

spec = importlib.util.spec_from_file_location(
    "m77", os.path.join(ROOT, "data", "77_p0234_analysis.py"))
m77 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m77)

WEIGHTS = os.path.join(ROOT, "data", "rmdb_audit", "RibonanzaNet.pt")
TOKENS = {c: i for i, c in enumerate("ACGU")}
CFG = SimpleNamespace(k=5, ninp=256, nlayers=9, nclass=2, ntoken=5,
                      nhead=8, dropout=0.0, pairwise_dimension=64,
                      use_triangular_attention=False)


def load_model():
    m = RibonanzaNet(CFG)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
    return m.cuda().eval()


@torch.no_grad()
def predict_2a3(model, seqs, batch_size=48):
    """{seq: np.ndarray 2A3}。官方口径:正反序列平均;原始输出通道 0=2A3。"""
    out = {}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        lens = [len(seqs[i]) for i in idx]
        L = max(lens)
        src = torch.full((len(idx), L), 4, dtype=torch.long)
        msk = torch.zeros((len(idx), L))
        for r, i in enumerate(idx):
            seq = seqs[i]
            src[r, :len(seq)] = torch.tensor([TOKENS[c] for c in seq])
            msk[r, :len(seq)] = 1
        src, msk = src.cuda(), msk.cuda()
        with torch.autocast("cuda", torch.float16):
            f = model(src, msk).float()
            sr = src.clone()
            for r, n in enumerate(lens):
                sr[r, :n] = sr[r, :n].flip(0)
            b = model(sr, msk).float()
            for r, n in enumerate(lens):
                b[r, :n] = b[r, :n].flip(0)
            pred = ((f + b) / 2)[:, :, 0]
        pred = pred.cpu().numpy()
        for r, i in enumerate(idx):
            out[seqs[i]] = pred[r, :lens[r]]
    return out


def auroc_signal(react, paired):
    """AUROC(反应性 ⇒ 未配对)。react [L], paired [L] bool。"""
    lab = ~paired
    n_pos, n_neg = int(lab.sum()), int((~lab).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(react)
    return float((r[lab].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    audit = json.load(open(os.path.join(SUR, "89_rnet_leakage_audit.json")))
    assert audit["verdict"] == "PASS", "泄露审计未通过,禁止筛选实验"

    model = load_model()
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2",
                                     "splits_v2.json")))
    data = load_s1(set(sp["test"]),
                   cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    stems = sorted(set(sp["test"]) & set(data))

    # ---- 阶段 1:GPU 推 RNet 2A3(逐链缓存设计序列反应性) ----
    t0 = time.time()
    react_cache, ref_ss, pools = {}, {}, {}
    for i, stem in enumerate(stems):
        p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
        if not os.path.exists(p) or len(data[stem]["seq"]) > 512:
            continue
        v = json.load(open(p))
        native, pool = v["native"], v["pool"]
        if len(native) < 15:
            continue
        pools[stem] = v
        ref_ss[stem] = m77.pairs_from_matrix(data[stem]["ss_pairs"])
        uniq = [s for s in {x for x in pool if set(x) <= set("ACGU")}
                if s not in react_cache]
        if uniq:
            react_cache.update(predict_2a3(model, uniq))
        if (i + 1) % 80 == 0:
            print(f"[90-gpu] {i+1}/{len(stems)} | {time.time()-t0:.0f}s",
                  flush=True)
    print(f"[90-gpu] done: {len(pools)} chains, "
          f"{len(react_cache)} unique designs | {time.time()-t0:.0f}s",
          flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- 阶段 2:CPU 回折 INF + 逐链指标(84 号同口径) ----
    per_chain = {}
    for j, stem in enumerate(sorted(pools)):
        v = pools[stem]
        native, pool, logp = v["native"], v["pool"], v["logp"]
        paired = np.zeros(len(native), dtype=bool)
        for a, b in ref_ss[stem]:
            if a < len(paired) and b < len(paired):
                paired[a] = paired[b] = True
        uniq = sorted({s for s in pool if set(s) <= set("ACGU")
                       and s in react_cache})
        structs = m77.fold_batch(uniq)
        pairmap = {s: m77.pairs_from_dbn(ss) for s, ss in zip(uniq, structs)
                   if ss is not None}
        good_idx = [k for k, s in enumerate(pool)
                    if s in pairmap and len(s) == len(native)]
        pool_g = [pool[k] for k in good_idx]
        rec = np.array([sum(1 for a, b in zip(s, native) if a == b) / len(native)
                        for s in pool_g])
        inf = np.array([m77.inf_scores(pairmap[s], ref_ss[stem])["inf"]
                        for s in pool_g])
        lp = np.array([logp[k] for k in good_idx], dtype=float)
        rnet = np.array([auroc_signal(react_cache[s], paired)
                         for s in pool_g])
        ok = ~np.isnan(rnet)
        rec, inf, lp, rnet = rec[ok], inf[ok], lp[ok], rnet[ok]
        if np.std(lp) == 0 or len(pool) < 50 or len(rec) < 50:
            continue
        rl = np.argsort(np.argsort(-lp))
        ri = np.argsort(np.argsort(-inf))
        rn = np.argsort(np.argsort(-rnet))
        per_chain[stem] = {"rec": rec, "inf": inf, "lp": lp, "rnet": rnet,
                           "cons2": rl + ri, "cons3": rl + ri + rn}
        if (j + 1) % 80 == 0:
            print(f"[90-cpu] {j+1}/{len(pools)} | {time.time()-t0:.0f}s",
                  flush=True)

    # ---- 阶段 3:预算曲线(84 号同款 schema) ----
    budgets = [0.01, 0.05, 0.10, 0.25, 0.50]
    strat_keys = {"random": None, "logP": "lp", "INF": "inf", "RNET": "rnet",
                  "consensus2": "cons2", "consensus3": "cons3",
                  "oracle": "rec"}
    curves = {}
    for strat, key in strat_keys.items():
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
            cur[f"budget_{b:.2f}"] = {
                "gain": float(np.mean(gains)),
                "ci95": [float(np.percentile(bs, 2.5)),
                         float(np.percentile(bs, 97.5))]}
        curves[strat] = cur

    # ---- 阶段 4:诊断 ----
    def col(name):
        return np.array([d[name].mean() for d in per_chain.values()])
    rhos_rr = [spearmanr(d["rnet"], d["rec"]).statistic
               for d in per_chain.values()]
    rhos_ri = [spearmanr(d["rnet"], d["inf"]).statistic
               for d in per_chain.values()]
    out = {
        "n_chains": len(per_chain),
        "curves": curves,
        "diagnostics": {
            "mean_rnet_auroc": float(np.mean(col("rnet"))),
            "mean_inf": float(np.mean(col("inf"))),
            "mean_chain_spearman_rnet_rec": float(np.nanmean(rhos_rr)),
            "frac_chain_spearman_rnet_rec_positive":
                float(np.mean(np.array(rhos_rr) > 0)),
            "mean_chain_spearman_rnet_inf": float(np.nanmean(rhos_ri)),
            "mean_chain_spearman_inf_rec": float(np.nanmean(
                [spearmanr(d["inf"], d["rec"]).statistic
                 for d in per_chain.values()])),
            "model_weights": "roos23/RibonanzaNet (HF mirror of Kaggle "
                             "RibonanzaNet.pt), 11.3M params",
            "rnet_signal": "AUROC(pred 2A3 reactivity => unpaired of input "
                           "SS); flip-averaged, channel 0",
            "audit": "survey/89_rnet_leakage_audit.json verdict=PASS",
        },
    }
    json.dump(out, open(os.path.join(SUR, "90_rnet_screen.json"), "w"),
              indent=1)
    for strat in ("random", "logP", "INF", "RNET", "consensus2", "consensus3",
                  "oracle"):
        c = curves[strat]
        line = " ".join(f"{b:.0%}:{c[f'budget_{b:.2f}']['gain']:+.4f}"
                        for b in budgets)
        print(f"[90] {strat:<11} {line}")
    d = out["diagnostics"]
    print(f"[90] meanRNET={d['mean_rnet_auroc']:.3f} "
          f"rho(rnet,rec)={d['mean_chain_spearman_rnet_rec']:.3f} "
          f"(pos {d['frac_chain_spearman_rnet_rec_positive']:.0%}) "
          f"rho(rnet,inf)={d['mean_chain_spearman_rnet_inf']:.3f} | "
          f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
