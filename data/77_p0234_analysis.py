#!/usr/bin/env python3
"""第 77 步：蓝图 P0-2/3/4 纯分析三件套（2026-08-18，analysis/12 §五）

P0-2 每核苷酸恢复率 vs 距配体距离曲线（Fig4b）——界面梯度是否存在
P0-3 条件保真度 INF：输入 SS vs 生成序列回折 SS（Fig4c，RNAbpFlow 式）
P0-4 置信度（log prob）-恢复率相关（Fig2h，Spearman+bootstrap CI）

相位：
  --phase pool   ens2 主配置 B 层 477 采样池（3 温度×100，39 号同协议/RNG）+ 逐序列
                 teacher-forced mean logP（集成概率平均）+ native logP
                 → survey/77_ens2_pool_btest/{stem}.json（GPU，~15min）
  --phase dist   P0-2：74 号元数据 115 界面链，逐残基距配体最近原子距离 × 逐位恢复率
                 （五方法池）分箱聚合 → survey/77_p02_distance_gradient.json（CPU）
  --phase inf    P0-3：RNAfold 回折每方法池（+native/随机对照）vs 输入 SS 的 INF/F1
                 → survey/77_p03_ss_fidelity.json（CPU，批量子进程）
  --phase conf   P0-4：链内/链间 Spearman + top-K enrichment
                 → survey/77_p04_confidence_recovery.json（CPU）
  --phase report 汇总打印三件套关键数字

环境：/root/miniconda3/envs/aptamer_eval/bin/python；GPU 用 1/2 号
（CUDA_VISIBLE_DEVICES 限制后脚本内 device=0）
"""
import os, sys, json, time, argparse, subprocess, tempfile, difflib
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100
POOL_DIR = os.path.join(SUR, "77_ens2_pool_btest")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
ENS_CKPTS = ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]   # 主配置 ens2-noiface（2026-08-19 对齐，原为含界面加权旧 ens2）

BASE_POOLS = {   # B 层各方法序列池读取器（返回 dict stem → list[str]，长度≠native 自动过滤）
    "RhoDesign":  lambda s: [l.strip() for t in TEMPS for l in open(f"{SUR}/rhodesign_out_btest_v2/{s}.{t}.seqs")],
    "gRNAde":     lambda s: [l.strip() for t in TEMPS for l in open(f"{SUR}/grnade_out_btest_v2/{s}.{t}.seqs")],
    "NA-MPNN":    lambda s: [l.strip() for l in open(f"{SUR}/nampnn_out_btest_v2/{s}.seqs")],
    "AlignIF":    lambda s: [l.strip() for l in open(f"{SUR}/alignif_out_btest_v2/{s}/out.fasta") if not l.startswith(">")],
}
TEMP_FILES = {   # 可分温度的方法 → t=0.1 池读取器（温度匹配对照）
    "RhoDesign": lambda s: [l.strip() for l in open(f"{SUR}/rhodesign_out_btest_v2/{s}.0.1.seqs")],
    "gRNAde":    lambda s: [l.strip() for l in open(f"{SUR}/grnade_out_btest_v2/{s}.0.1.seqs")],
}

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m35 = m74.m35


def test_stems():
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    return list(sp2["test"])


# ================================================================ pool（GPU）
def phase_pool(gpu_note="GPU"):
    import torch
    import torch.nn.functional as F
    m39 = _imp("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
    from PACoAptamer.data_utils import load_s1

    device = 0
    torch.manual_seed(42)
    models, alphabets = [], []
    for n in ENS_CKPTS:
        p = os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, a = m74.load_model(p, device)
        models.append(m); alphabets.append(a)
    alphabet = alphabets[0]

    stems = test_stems()
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    stems = [s for s in stems if s in data]
    os.makedirs(POOL_DIR, exist_ok=True)

    @torch.no_grad()
    def score_seqs(seqs, d, chunk=100):
        """逐序列 teacher-forced mean logP；集成 = softmax 概率平均（与集成采样同语义）"""
        L = len(d['seq'])
        coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        bos = alphabet.get_idx("<cath>")
        out = []
        for s0 in range(0, len(seqs), chunk):
            batch = seqs[s0:s0 + chunk]
            B = len(batch)
            prev = torch.zeros(B, L, dtype=torch.long, device=device)
            tgt = torch.zeros(B, L, dtype=torch.long, device=device)
            for bi, sq in enumerate(batch):
                toks = [alphabet.get_idx(c) for c in sq]
                prev[bi, 0] = bos
                prev[bi, 1:] = torch.tensor(toks[:-1], device=device)
                tgt[bi] = torch.tensor(toks, device=device)
            probs = None
            for m in models:
                eo = m.forward_embedding(coords, ss, lig, ligm, pmask, conf)
                eoB = {"encoder_out": [eo["encoder_out"][0].repeat(1, B, 1)],
                       "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(B, 1)],
                       "encoder_embedding": [], "encoder_states": []}
                logits, _ = m.decoder(prev, encoder_out=eoB)     # (B, V, L)
                p = F.softmax(logits, dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(models)
            lp = torch.log(probs + 1e-12).gather(1, tgt.unsqueeze(1)).squeeze(1)  # (B, L)
            out.extend(lp.mean(dim=1).float().cpu().numpy().tolist())
        return out

    t0 = time.time()
    n_done = 0
    for stem in stems:
        out_p = os.path.join(POOL_DIR, stem + ".json")
        if os.path.exists(out_p):
            continue
        d = data[stem]
        if len(d['seq']) > 512:
            continue
        native = d['seq']
        try:
            pool = []
            for t in TEMPS:
                seqs = m39.sample_chain_ens(models, alphabet, d, device, t, N_PER_TEMP)
                pool.extend(seqs)
            logp = score_seqs(pool, d)
            logp_native = score_seqs([native], d)[0]
        except RuntimeError as e:
            print(f"  [pool 跳过 {stem}: {e}]", flush=True)
            continue
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        json.dump({"len": len(native), "native": native, "pool": pool, "logp": logp,
                   "logp_native": logp_native, "n_per_temp": N_PER_TEMP,
                   "recovery_best": max(recs),
                   "recovery_t01_mean": float(np.mean(recs[:N_PER_TEMP]))},
                  open(out_p, "w"))
        n_done += 1
        if n_done % 50 == 0:
            print(f"[pool] {n_done} 链完成 | {time.time()-t0:.0f}s", flush=True)
    # 复现校验（39 号 ens2 full：best 0.7463 / t01 0.5955）
    rb, rt = [], []
    for stem in stems:
        p = os.path.join(POOL_DIR, stem + ".json")
        if os.path.exists(p):
            v = json.load(open(p))
            rb.append(v["recovery_best"]); rt.append(v["recovery_t01_mean"])
    print(f"[pool] n={len(rb)} best={np.mean(rb):.4f} t01={np.mean(rt):.4f}"
          f"（39 号记录 0.7463/0.5955）", flush=True)


# ================================================================ P0-2 dist
BINS = [(0.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 1e9)]
BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]


def load_paco_pool(stem):
    p = os.path.join(POOL_DIR, stem + ".json")
    return json.load(open(p))["pool"] if os.path.exists(p) else None


def load_native_seq(stem, data):
    return data[stem]['seq'] if stem in data else None


def phase_dist():
    from scipy.spatial import cKDTree
    from PACoAptamer.data_utils import load_s1
    stems = test_stems()
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    usable = {k: v for k, v in meta.items()
              if v and v.get("usable") and 0 < v["n_iface"] <= 50}

    # 每残基距配体最近原子距离（74 号同款映射：35 号提取 → native 序坐标系）
    dist_native = {}
    for stem, mv in usable.items():
        pdb_id, ch = stem.rsplit("_", 1)
        native = load_native_seq(stem, data)
        if native is None:
            continue
        residues, seq35, _ = m35.extract_rna(pdb_id, ch)
        if seq35 is None or len(residues) != len(seq35):
            continue
        if seq35 == native:
            mapping = list(range(len(native)))
        else:
            sm = difflib.SequenceMatcher(None, seq35, native, autojunk=False)
            m2n = {}
            ok = True
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    m2n[a + k] = b + k
            for i in range(len(seq35)):
                if i not in m2n:
                    ok = False; break
            if not ok or len(m2n) != len(native):
                continue
            mapping = [m2n[i] for i in range(len(seq35))]
        tree_lig = cKDTree(np.array(mv["lig_coords"], dtype=np.float32))
        dists = np.full(len(native), np.nan)
        for i, (_, _, _, atoms) in enumerate(residues):
            pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
            if len(pts):
                dd, _ = tree_lig.query(pts, k=1)
                dists[mapping[i]] = float(dd.min())
        dist_native[stem] = {"native": native, "dist": dists,
                             "n_iface_meta": mv["n_iface"]}
    print(f"[dist] 距离可算链 {len(dist_native)}/{len(usable)}")

    # 每方法逐位恢复率 × 距离分箱
    methods = ["PACo_ens2"] + list(BASE_POOLS)
    per_chain = {m: {} for m in methods}          # stem → {bin: (mean_rec, n_pos, n_seq)}
    for stem, dv in dist_native.items():
        native, dists = dv["native"], dv["dist"]
        valid = np.array([c != "N" and np.isfinite(dists[i])
                          for i, c in enumerate(native)])
        for m in methods:
            if m == "PACo_ens2":
                pool = load_paco_pool(stem)
            else:
                try:
                    pool = BASE_POOLS[m](stem)
                except FileNotFoundError:
                    pool = None
            if not pool:
                continue
            seqs = [s for s in pool if len(s) == len(native)]
            if len(seqs) < 5:
                continue
            mat = np.array([list(s) for s in seqs])            # (n, L)
            rec = (mat == np.array(list(native))).mean(axis=0)  # (L,)
            rec_by_bin, n_by_bin = {}, {}
            for (lo, hi), nm in zip(BINS, BIN_NAMES):
                sel = valid & (dists >= lo) & (dists < hi)
                if sel.sum() > 0:
                    rec_by_bin[nm] = float(rec[sel].mean())
                    n_by_bin[nm] = int(sel.sum())
            glob = float(rec[valid].mean())   # 本链全有效位均值（归一化基准）
            per_chain[m][stem] = {"rec": rec_by_bin, "n": n_by_bin,
                                  "n_seq": len(seqs), "glob": glob}

    def boot_ci(x, n=2000, seed=0):
        if len(x) < 3:
            return None
        rng = np.random.RandomState(seed)
        m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    out = {"n_iface_chains_dist": len(dist_native), "methods": {}}
    for m in methods:
        rows = {}
        chains = per_chain[m]
        for nm in BIN_NAMES:
            vals = [c["rec"][nm] for c in chains.values() if nm in c["rec"]]
            if vals:
                rows[nm] = {"n_chains": len(vals), "mean": float(np.mean(vals)),
                            "ci95": boot_ci(vals), "median": float(np.median(vals))}
            # 归一化梯度：bin − 本链全有效位均值（去掉方法强弱，只留形状）
            nvals = [c["rec"][nm] - c["glob"] for c in chains.values()
                     if nm in c["rec"]]
            if nvals:
                rows[nm]["norm_mean"] = float(np.mean(nvals))
                rows[nm]["norm_ci95"] = boot_ci(nvals)
        # 界面(≤5) − 远端(>12) 逐链配对差
        difs = [c["rec"]["≤5Å(界面)"] - c["rec"][">20Å"] for c in chains.values()
                if "≤5Å(界面)" in c["rec"] and ">20Å" in c["rec"]]
        rows["iface_minus_gt20"] = {"n": len(difs), "mean": float(np.mean(difs)),
                                    "ci95": boot_ci(difs)} if difs else None
        rows["n_chains_total"] = len(chains)
        out["methods"][m] = rows
    json.dump(out, open(os.path.join(SUR, "77_p02_distance_gradient.json"), "w"), indent=1)
    for m in methods:
        r = out["methods"][m]
        line = " | ".join(f"{nm} {r[nm]['mean']:.3f}({r[nm]['n_chains']})"
                          for nm in BIN_NAMES if nm in r)
        print(f"[P0-2] {m}: {line}")
        line2 = " | ".join(f"{nm} {r[nm].get('norm_mean', float('nan')):+.3f}"
                           for nm in BIN_NAMES if nm in r)
        print(f"       归一化(bin−链均值): {line2}")
        if r.get("iface_minus_gt20"):
            print(f"       界面−远端 Δ={r['iface_minus_gt20']['mean']:+.4f}")


# ================================================================ P0-3 INF
def pairs_from_dbn(dbn):
    pairs = set()
    stacks = {"(": [], "[": [], "{": [], "<": []}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, c in enumerate(dbn):
        if c in stacks:
            stacks[c].append(i)
        elif c in close:
            if stacks[close[c]]:
                j = stacks[close[c]].pop()
                pairs.add((min(i, j), max(i, j)))
    return pairs


def pairs_from_matrix(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return {(int(i), int(j)) for i, j, v in zip(*iu, mat[iu]) if v}


def fold_batch(seqs):
    """RNAfold 批量回折（单子进程，逐行输入）→ dot-bracket 列表"""
    if not seqs:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        f.write("\n".join(seqs) + "\n")
        fn = f.name
    try:
        r = subprocess.run([RNAFOLD, "--noPS"], stdin=open(fn),
                           capture_output=True, text=True, timeout=600)
        struct_lines = [l for l in r.stdout.splitlines()
                        if l and l[0] in ".([{<"]
        structs = []
        import re
        for l in struct_lines:
            m = re.match(r"([\.\(\)\[\]\{\}\<\>]+)", l.strip())
            if m:
                structs.append(m.group(1))
        if len(structs) != len(seqs):
            print(f"    [fold 数量不匹配 {len(structs)}/{len(seqs)}]", flush=True)
            return []
        return structs
    finally:
        os.unlink(fn)


def inf_scores(pred_pairs, ref_pairs):
    NR, NP, TP = len(ref_pairs), len(pred_pairs), len(pred_pairs & ref_pairs)
    if NR == 0 and NP == 0:
        inf = 1.0
    else:
        inf = TP / np.sqrt(NR * NP) if NR * NP > 0 else 0.0
    ppv = TP / NP if NP else (1.0 if NR == 0 else 0.0)
    sen = TP / NR if NR else (1.0 if NP == 0 else 0.0)
    f1 = 2 * TP / (NR + NP) if (NR + NP) else 1.0
    return {"inf": inf, "ppv": ppv, "sens": sen, "f1": f1}


def _inf_work_one(args):
    """worker（模块级以便 pickle）：链上所有方法池合并去重 → 一次 RNAfold → 逐方法切片算 INF"""
    import zlib
    stem, native, ref_pairs = args
    rng = np.random.RandomState(zlib.crc32(stem.encode()) % (2**31))
    pools = {}
    tp_ = load_paco_pool(stem)
    if tp_:
        pools["PACo_ens2"] = tp_
    for m, fn in BASE_POOLS.items():
        try:
            pools[m] = [s for s in fn(stem) if len(s) == len(native)]
        except FileNotFoundError:
            pass
    # t=0.1 子集 = 各池前 100（温度顺序 0.1/0.5/1.0），切片复用不重复回折
    for m in ("PACo_ens2", "RhoDesign", "gRNAde"):
        if m in pools and len(pools[m]) >= N_PER_TEMP:
            pools[m + "_t01"] = pools[m][:N_PER_TEMP]
    pools["random"] = ["".join(rng.choice(list("ACGU"), len(native)))
                       for _ in range(50)]
    pools["native"] = [native]
    # 全链去重折叠（t=0.1 采样重复率高，收益大）
    uniq = sorted({s for seqs in pools.values() for s in seqs if set(s) <= set("ACGU")})
    if not uniq:
        return stem, {}
    structs = fold_batch(uniq)
    if not structs:
        return stem, {}
    pairmap = {s: pairs_from_dbn(ss) for s, ss in zip(uniq, structs)}
    out = {}
    for m, seqs in pools.items():
        sc = []
        for s in seqs:
            if s not in pairmap:
                continue
            r = inf_scores(pairmap[s], ref_pairs)
            r["ident"] = sum(1 for a, b in zip(s, native) if a == b) / len(native)
            sc.append(r)
        if sc:
            out[m] = sc
    return stem, out


def phase_inf():
    from PACoAptamer.data_utils import load_s1
    from concurrent.futures import ProcessPoolExecutor
    stems = test_stems()
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    # 坑：load_s1 命中缓存时忽略 subset 过滤（缓存含全部 6338 链）→ 必须按 splits 键过滤
    args_list = [(s, data[s]['seq'], pairs_from_matrix(data[s]['ss_pairs']))
                 for s in sorted(set(stems) & set(data))
                 if len(data[s]['seq']) <= 512]

    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, rec) in enumerate(ex.map(_inf_work_one, args_list, chunksize=4)):
            if rec:
                for m, sc in rec.items():
                    per_chain.setdefault(m, {})[stem] = sc
            if (i + 1) % 50 == 0:
                print(f"[inf] {i+1}/{len(args_list)} | {time.time()-t0:.0f}s", flush=True)

    def boot_ci(x, n=2000, seed=0):
        if len(x) < 3:
            return None
        r = np.random.RandomState(seed)
        m = [r.choice(x, len(x), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    out = {"methods": {}}
    for m, chains in per_chain.items():
        agg = {}
        for key in ("inf", "f1", "ppv", "sens"):
            vals = [np.mean([s[key] for s in scs]) for scs in chains.values()]
            agg[key] = {"n_chains": len(vals), "mean": float(np.mean(vals)),
                        "ci95": boot_ci(vals), "median": float(np.median(vals))}
        agg["frac_design_inf_ge0.9"] = float(np.mean(
            [s["inf"] >= 0.9 for scs in chains.values() for s in scs]))
        # 诊断：按与 native 序列相似度分箱的 INF（MFE 脆性 vs 条件跟随性分解）
        id_edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
        by_ident = {}
        for lo, hi in zip(id_edges[:-1], id_edges[1:]):
            vals = [s["inf"] for scs in chains.values() for s in scs
                    if lo <= s.get("ident", -1) < hi]
            if len(vals) >= 20:
                by_ident[f"{lo:.1f}-{hi:.1f}"] = {
                    "n": len(vals), "inf_mean": float(np.mean(vals))}
        agg["inf_by_identity"] = by_ident
        # 相似度与 INF 的相关（设计层 Spearman）
        ids = [s["ident"] for scs in chains.values() for s in scs]
        infs = [s["inf"] for scs in chains.values() for s in scs]
        if len(ids) > 30 and np.std(ids) > 0:
            from scipy.stats import spearmanr
            agg["spearman_ident_vs_inf"] = float(spearmanr(ids, infs).statistic)
        out["methods"][m] = agg
        print(f"[P0-3] {m}: INF {agg['inf']['mean']:.3f} F1 {agg['f1']['mean']:.3f}"
              f" (n={agg['inf']['n_chains']})", flush=True)
    json.dump(out, open(os.path.join(SUR, "77_p03_ss_fidelity.json"), "w"), indent=1)


# ================================================================ P0-4 conf
def phase_conf():
    from scipy.stats import spearmanr
    stems = test_stems()
    chains = {}
    for stem in stems:
        p = os.path.join(POOL_DIR, stem + ".json")
        if os.path.exists(p):
            chains[stem] = json.load(open(p))
    print(f"[conf] 链 {len(chains)}")

    # (a) 链内：每链 300 设计的 Spearman(logP, recovery)
    in_rho = []
    for stem, v in chains.items():
        lp, rc = v["logp"], []
        for s in v["pool"]:
            rc.append(sum(1 for a, b in zip(s, v["native"]) if a == b) / len(v["native"]))
        if len(lp) >= 30 and np.std(lp) > 0 and np.std(rc) > 0:
            r = spearmanr(lp, rc).statistic
            if np.isfinite(r):
                in_rho.append(float(r))

    # (b) 链间：mean logP vs recovery（t01 均值 / best）
    x = [float(np.mean(v["logp"])) for v in chains.values()]
    y_t = [v["recovery_t01_mean"] for v in chains.values()]
    y_b = [v["recovery_best"] for v in chains.values()]

    def spear_boot(xv, yv, n=5000, seed=0):
        r = float(spearmanr(xv, yv).statistic)
        rng = np.random.RandomState(seed)
        xv, yv = np.asarray(xv), np.asarray(yv)
        bs = []
        for _ in range(n):
            idx = rng.choice(len(xv), len(xv), replace=True)
            if np.std(xv[idx]) > 0 and np.std(yv[idx]) > 0:
                rr = spearmanr(xv[idx], yv[idx]).statistic
                if np.isfinite(rr):
                    bs.append(rr)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
        return r, ci

    r_t, ci_t = spear_boot(x, y_t)
    r_b, ci_b = spear_boot(x, y_b)

    # (c) 置信度排序实用价值：top-10%/top-1 平均恢复率 vs 池均值
    enr10, enr1 = [], []
    for v in chains.values():
        lp = np.array(v["logp"])
        rc = np.array([sum(1 for a, b in zip(s, v["native"]) if a == b) / len(v["native"])
                       for s in v["pool"]])
        order = np.argsort(-lp)
        k = max(1, len(lp) // 10)
        enr10.append(float(rc[order[:k]].mean() - rc.mean()))
        enr1.append(float(rc[order[0]] - rc.mean()))

    def boot_ci(x, n=2000, seed=0):
        r = np.random.RandomState(seed)
        m = [r.choice(x, len(x), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    out = {
        "n_chains": len(chains),
        "within_chain_spearman": {
            "n": len(in_rho), "mean": float(np.mean(in_rho)),
            "median": float(np.median(in_rho)), "ci95": boot_ci(in_rho),
            "frac_positive": float(np.mean(np.array(in_rho) > 0))},
        "chain_level_spearman": {
            "mean_logp_vs_recovery_t01": {"rho": r_t, "ci95": ci_t},
            "mean_logp_vs_recovery_best": {"rho": r_b, "ci95": ci_b}},
        "logp_rank_enrichment": {
            "top10pct_minus_pool_mean": {"mean": float(np.mean(enr10)), "ci95": boot_ci(enr10)},
            "top1_minus_pool_mean": {"mean": float(np.mean(enr1)), "ci95": boot_ci(enr1)}},
    }
    json.dump(out, open(os.path.join(SUR, "77_p04_confidence_recovery.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ================================================================ report
def phase_report():
    for name, fn in (("P0-2 距离梯度", "77_p02_distance_gradient.json"),
                     ("P0-3 SS 保真度", "77_p03_ss_fidelity.json"),
                     ("P0-4 置信度", "77_p04_confidence_recovery.json")):
        p = os.path.join(SUR, fn)
        if os.path.exists(p):
            print(f"\n===== {name} =====")
            print(json.dumps(json.load(open(p)), indent=1)[:3500])
        else:
            print(f"\n===== {name}（缺 {fn}）=====")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    choices=['pool', 'dist', 'inf', 'conf', 'report'])
    args = ap.parse_args()
    dict(pool=phase_pool, dist=phase_dist, inf=phase_inf,
         conf=phase_conf, report=phase_report)[args.phase]()


if __name__ == "__main__":
    main()
