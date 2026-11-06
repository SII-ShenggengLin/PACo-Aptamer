#!/usr/bin/env python3
"""第 79 步：AptaDB 回顾性验证（P1-1，蓝图 §五 Fig6d——"计算版湿实验"）

对已知 Kd 的适配体（AptaDB，RNA 2024）做模型置信度的回顾性分析：
  a. 序列先验：logP(适配体 | 其预测结构) vs 随机序列对照（同结构同 SS / 同结构各自 SS）
  b. 家族内 Kd 相关：同靶标家族内 Spearman(logP, −log10 Kd)
  c. 最优 Kd 排名检验：家族内最佳亲和力适配体的模型排名百分位（vs 0.5 零假设）
  d. 全库分层/预算曲线（gRNAde 图6 式，样本量允许时）

相位：
  --phase clean   过滤清洗（RNA-only / ACGU / 长度 / Kd 解析 / 去重）→ 79_aptadb_clean.json + fold 输入 fasta
  --phase leak    cd-hit-est 0.8 联聚类 vs 训练序列库 → 泄漏标注（exact + 簇级）
  --phase fold    RhoFold+ 批量预测（GPU 1/2 并行，15 号同款命令）→ 79_rhofold/{key}.pdb
  --phase score   ens2 teacher-forced logP（适配体 + 双随机对照 + 错配结构对照）→ 79_aptadb_scored.json
  --phase report  a-d 分析 → 79_aptadb_retro.json

环境：/root/miniconda3/envs/aptamer_eval/bin/python；GPU 用 1/2 号
"""
import os, sys, json, time, argparse, re, subprocess, tempfile
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

CLEAN_J = os.path.join(SUR, "79_aptadb_clean.json")
FOLD_DIR = os.path.join(SUR, "79_rhofold")
SCORED_J = os.path.join(SUR, "79_aptadb_scored.json")
REPORT_J = os.path.join(SUR, "79_aptadb_retro.json")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
RHOFOLD_REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
ENS_CKPTS = ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]   # 主配置 ens2-noiface（2026-08-19 对齐）
MIN_LEN, MAX_LEN = 15, 300        # 模型 512 上限；SELEX 适配体典型 20-80，300 留余量
N_RANDOM = 10

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m77 = _imp("m77", os.path.join(DATA_DIR, "77_p0234_analysis.py"))   # 复用 fold_batch / pairs_from_dbn


# ---------------------------------------------------------------- clean
KD_UNITS = {"pM": 1e-3, "nM": 1.0, "µM": 1e3, "μM": 1e3, "uM": 1e3, "mM": 1e6, "M": 1e9}


def parse_kd_nM(raw):
    """'120 pM'/'35.6 nM'/'1.3 μM'/'10-20 nM'/'<5 nM'/... → nM float（区间取中点），失败 None"""
    if not raw:
        return None
    s = raw.replace("Kd", "").replace("kD", "").replace("=", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–~to]{1,3}\s*(\d+(?:\.\d+)?)\s*(pM|nM|µM|μM|uM|mM)", s)
    if m:
        lo, hi, u = float(m.group(1)), float(m.group(2)), m.group(3)
        return (lo + hi) / 2 * KD_UNITS[u]
    m = re.search(r"[<≤>≥约~]?\s*(\d+(?:\.\d+)?)\s*(pM|nM|µM|μM|uM|mM)\b", s)
    if m:
        return float(m.group(1)) * KD_UNITS[m.group(2)]
    return None


def phase_clean():
    raw = json.load(open(os.path.join(SUR, "78_aptadb.json")))
    stats = {"total": len(raw)}
    entries = []
    for iid, v in raw.items():
        seq = (v.get("seq") or "").upper().replace("T", "N")   # DNA 碱基 T → 标记为 N
        chem = (v.get("chemistry") or "").strip().upper()
        if chem and "RNA" not in chem:                          # DNA/LNA 等剔除
            stats["non_rna"] = stats.get("non_rna", 0) + 1
            continue
        if not seq or len(seq) < MIN_LEN or len(seq) > MAX_LEN:
            stats["len_out"] = stats.get("len_out", 0) + 1
            continue
        if not set(seq) <= set("ACGU"):
            stats["non_acgu"] = stats.get("non_acgu", 0) + 1
            continue
        kd = parse_kd_nM(v.get("affinity"))
        entries.append({
            "interaction_id": int(iid),
            "key": f"apta{iid}",
            "seq": seq,
            "target_uid": v.get("target_uid"),
            "target_chemistry": v.get("target_chemistry"),
            "target_name": v.get("target_name"),
            "affinity_raw": v.get("affinity"),
            "kd_nM": kd,
            "pubmed_id": v.get("pubmed_id"),
        })
    stats["kept"] = len(entries)
    stats["with_kd"] = sum(1 for e in entries if e["kd_nM"] is not None)
    # 同序列同靶标去重（保留首个；多条不同 Kd 记录标注 n_dup）
    seen = {}
    dedup = []
    for e in entries:
        k = (e["seq"], e["target_uid"])
        if k in seen:
            seen[k]["n_dup"] = seen[k].get("n_dup", 1) + 1
            if e["kd_nM"] and seen[k]["kd_nM"] is None:
                seen[k]["kd_nM"] = e["kd_nM"]
                seen[k]["affinity_raw"] = e["affinity_raw"]
            continue
        seen[k] = e
        dedup.append(e)
    stats["dedup"] = len(dedup)
    fam = {}
    for e in dedup:
        fam.setdefault(e["target_uid"], []).append(e)
    stats["n_targets"] = len(fam)
    stats["targets_ge3_kd"] = sum(1 for v in fam.values()
                                  if sum(1 for x in v if x["kd_nM"]) >= 3)
    stats["targets_ge2_kd"] = sum(1 for v in fam.values()
                                  if sum(1 for x in v if x["kd_nM"]) >= 2)
    json.dump({"stats": stats, "entries": dedup}, open(CLEAN_J, "w"), indent=1)
    # RhoFold 输入 fasta（唯一序列）
    uniq = {}
    for e in dedup:
        uniq.setdefault(e["seq"], e["key"])
    os.makedirs(FOLD_DIR, exist_ok=True)
    with open(os.path.join(FOLD_DIR, "aptadb_input.fa"), "w") as f:
        for s, k in uniq.items():
            f.write(f">{k}\n{s}\n")
    stats["unique_seqs"] = len(uniq)
    json.dump(stats, open(os.path.join(FOLD_DIR, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- leak
def phase_leak():
    from subprocess import run
    d = json.load(open(CLEAN_J))
    entries = d["entries"]
    lib = json.load(open(os.path.join(DATA_DIR, "splits_v2", "seqs_all.json")))
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    stem2set = {}
    for s in ("train", "val", "test", "independent_test", "zero_shot_targets"):
        for stem in sp[s]:
            stem2set[stem] = s
    lib_seqs = {stem: seq for stem, seq in lib.items() if stem in stem2set}
    # 合并 fasta：适配体 + 库
    merged = os.path.join(SUR, "79_leak_in.fa")
    with open(merged, "w") as f:
        seen_s = set()
        for e in entries:
            if e["seq"] not in seen_s:
                seen_s.add(e["seq"])
                f.write(f">{e['key']}\n{e['seq']}\n")
        for stem, s in lib_seqs.items():
            f.write(f">{stem}\n{s}\n")
    out_clstr = os.path.join(SUR, "79_leak_out.fa.clstr")
    r = run(["cd-hit-est", "-i", merged, "-o", os.path.join(SUR, "79_leak_out.fa"),
             "-c", "0.8", "-n", "8", "-d", "0", "-T", "16", "-M", "0"],
            capture_output=True, text=True)
    print(r.stdout[-500:] if r.returncode == 0 else r.stderr[-800:])
    # 解析 clstr：簇内混合 apta×library 成员 → 标注
    clusters, cur = {}, None
    for line in open(out_clstr):
        if line.startswith(">Cluster"):
            cur = line.strip()
            clusters[cur] = []
        else:
            m = re.search(r">(.*?)\.\.\.", line)
            if m and cur:
                clusters[cur].append(m.group(1))
    seq2cluster = {}
    for cname, members in clusters.items():
        for m in members:
            seq2cluster[m] = cname
    # exact 匹配
    lib_seq_index = {}
    for stem, s in lib_seqs.items():
        lib_seq_index.setdefault(s, []).append(stem)
    n_exact_train = n_cluster_train = 0
    for e in entries:
        e["leak"] = {"exact": None, "cluster_train": False, "cluster_eval": False}
        hits = lib_seq_index.get(e["seq"])
        if hits:
            sets = {stem2set[h] for h in hits}
            e["leak"]["exact"] = sorted(sets)
        c = seq2cluster.get(e["key"])
        if c:
            lib_m = [m for m in clusters[c] if not m.startswith("apta")]
            sets = {stem2set[m] for m in lib_m if m in stem2set}
            if "train" in sets:
                e["leak"]["cluster_train"] = True
            if sets & {"test", "independent_test", "zero_shot_targets", "val"}:
                e["leak"]["cluster_eval"] = True
        if e["leak"]["exact"] and "train" in e["leak"]["exact"]:
            n_exact_train += 1
        if e["leak"]["cluster_train"]:
            n_cluster_train += 1
    d["leak_summary"] = {
        "n_entries": len(entries),
        "n_exact_train": n_exact_train,
        "n_cluster_train_0.8": n_cluster_train,
        "n_clean": sum(1 for e in entries
                       if not e["leak"]["cluster_train"]
                       and not (e["leak"]["exact"] and "train" in e["leak"]["exact"])),
    }
    json.dump(d, open(CLEAN_J, "w"), indent=1)
    print(json.dumps(d["leak_summary"], indent=1))


# ---------------------------------------------------------------- fold
def phase_fold():
    from concurrent.futures import ThreadPoolExecutor
    d = json.load(open(CLEAN_J))
    os.makedirs(FOLD_DIR, exist_ok=True)
    uniq = {}
    for e in d["entries"]:
        uniq.setdefault(e["seq"], e["key"])
    items = list(uniq.items())

    def run_one(item, gpu):
        seq, key = item          # uniq 映射 seq → key
        out_pdb = os.path.join(FOLD_DIR, key + ".pdb")
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 1000:
            return "cached"
        fa = os.path.join(FOLD_DIR, key + ".fa")
        with open(fa, "w") as f:
            f.write(f">{key}\n{seq}\n")
        wd = os.path.join(FOLD_DIR, "tmp_" + key)
        cmd = [RHOFOLD_PY, "inference.py", "--input_fas", fa, "--output_dir", wd,
               "--device", f"cuda:{gpu}", "--single_seq_pred", "True",
               "--relax_steps", "0", "--ckpt", f"{RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
        try:
            r = subprocess.run(cmd, cwd=RHOFOLD_REPO, capture_output=True, timeout=600)
            pred = os.path.join(wd, "unrelaxed_model.pdb")
            if r.returncode != 0 or not os.path.exists(pred):
                return "fail"
            os.rename(pred, out_pdb)
            for f in os.listdir(wd):
                os.remove(os.path.join(wd, f))
            os.rmdir(wd)
            return "ok"
        except Exception:
            return "fail"

    t0 = time.time()
    stats = {"ok": 0, "cached": 0, "fail": 0}
    lock_gpus = [1, 2]
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, it, lock_gpus[i % 2]) for i, it in enumerate(items)]
        for i, f in enumerate(futs):
            stats[f.result()] = stats.get(f.result(), 0) + 1
            if (i + 1) % 50 == 0:
                print(f"[fold] {i+1}/{len(items)} {stats} | {time.time()-t0:.0f}s", flush=True)
    print(f"[fold] 完成 {stats} | {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- score
BASE_ATOMS = ["C4'", "C1'", "N1", "C2", "C5'", "O5'", "P", "C2'", "C3'"]


def coords9_from_pdb(path):
    """RhoFold+ PDB → (L, 9, 3)；第 3 槽位一律取字面 N1 原子——与训练约定一致
    （data_utils.to_paco9 的 PACO9 索引 2 = G27 字面 "N1"，对所有碱基含嘌呤），
    曾误用 N9（嘌呤糖苷键原子）→ 碱基朝向坐标系偏离训练分布，2026-08-19 修复。缺原子=NaN"""
    res = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        base = line[17:20].strip().upper()
        base = base[-1] if base and base[-1] in "ACGU" else "N"
        key = (line[21:22].strip() or "A", line[22:26].strip(), line[26:27].strip())
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        res.setdefault(key, {"base": base, "atoms": {}})["atoms"][atom] = xyz
    if not res:
        return None
    keys = sorted(res, key=lambda k: (k[0], int(k[1]), k[2]))
    out = np.full((len(keys), 9, 3), np.nan, dtype=np.float32)
    for i, k in enumerate(keys):
        r = res[k]
        for j, an in enumerate(BASE_ATOMS):
            if an in r["atoms"]:
                out[i, j] = r["atoms"][an]
    return out


def fold_ss(seqs):
    structs = m77.fold_batch(seqs)
    return structs


def phase_score():
    import torch
    import torch.nn.functional as F
    from PACoAptamer.data_utils import ss_to_pairs
    device = 0
    models, alpha = [], None
    for n in ENS_CKPTS:
        m, a = m74.load_model(os.path.join(ROOT, "models", "PACoAptamer", n + ".pth"), device)
        models.append(m); alpha = a
    alphabet = alpha
    bos = alphabet.get_idx("<cath>")

    @torch.no_grad()
    def logp_batch(seqs, coords9, ss_pairs):
        L = coords9.shape[0]
        coords = torch.tensor(coords9, dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(ss_pairs, dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        B = len(seqs)
        prev = torch.zeros(B, L, dtype=torch.long, device=device)
        tgt = torch.zeros(B, L, dtype=torch.long, device=device)
        for bi, sq in enumerate(seqs):
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
            logits, _ = m.decoder(prev, encoder_out=eoB)
            p = F.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
        probs = probs / len(models)
        lp = torch.log(probs + 1e-12).gather(1, tgt.unsqueeze(1)).squeeze(1)
        return lp.mean(dim=1).float().cpu().numpy()

    d = json.load(open(CLEAN_J))
    entries = d["entries"]
    scored = {}
    if os.path.exists(SCORED_J):
        scored = json.load(open(SCORED_J)).get("entries", {})
    t0 = time.time()
    n_done = 0
    for e in entries:
        key = e["key"]
        if key in scored:
            continue
        pdb_p = os.path.join(FOLD_DIR, key + ".pdb")
        if not os.path.exists(pdb_p):
            continue
        coords9 = coords9_from_pdb(pdb_p)
        if coords9 is None or coords9.shape[0] != len(e["seq"]) or \
                np.isnan(coords9[:, :3, :]).all():
            scored[key] = {"failed": "coords"}
            continue
        rng = np.random.RandomState(zlib_crc(key))
        rand = ["".join(rng.choice(list("ACGU"), len(e["seq"]))) for _ in range(N_RANDOM)]
        # SS：适配体自身 RNAfold；随机对照各自 RNAfold
        structs = fold_ss([e["seq"]] + rand)
        if not structs or len(structs) != N_RANDOM + 1:
            scored[key] = {"failed": "fold"}
            continue
        try:
            lp_self = float(logp_batch([e["seq"]], coords9, ss_to_pairs(structs[0]))[0])
            lp_randA = logp_batch(rand, coords9, ss_to_pairs(structs[0]))       # 同 SS
            # 对照 B：各随机序列用各自 RNAfold SS（配对矩阵不同 → 逐条算）
            lp_randB = [float(logp_batch([r], coords9, ss_to_pairs(s))[0])
                        for r, s in zip(rand, structs[1:])]
        except RuntimeError as err:
            scored[key] = {"failed": f"model:{err}"}
            continue
        scored[key] = {"logp_self": lp_self,
                       "logp_self_per_nt": lp_self / len(e["seq"]),
                       "logp_rand_sameSS": [float(x) for x in lp_randA],
                       "logp_rand_ownSS": lp_randB,
                       "len": len(e["seq"])}
        n_done += 1
        if n_done % 50 == 0:
            print(f"[score] {n_done} 新打分 | {time.time()-t0:.0f}s", flush=True)
    json.dump({"entries": scored}, open(SCORED_J, "w"))
    print(f"[score] 完成：scored {sum(1 for v in scored.values() if 'logp_self' in v)}"
          f" / failed {sum(1 for v in scored.values() if v.get('failed'))}")


def zlib_crc(s):
    import zlib
    return zlib.crc32(s.encode()) % (2**31)


# ---------------------------------------------------------------- report
def phase_report():
    from scipy.stats import spearmanr, wilcoxon
    d = json.load(open(CLEAN_J))
    sc = json.load(open(SCORED_J))["entries"]
    entries = []
    for e in d["entries"]:
        s = sc.get(e["key"])
        if s and "logp_self" in s:
            ee = dict(e)
            ee["logp_self"] = s["logp_self"]
            ee["logp_per_nt"] = s["logp_self_per_nt"]
            ee["logp_randA_mean"] = float(np.mean(s["logp_rand_sameSS"]))
            ee["logp_randB_mean"] = float(np.mean(s["logp_rand_ownSS"]))
            ee["len"] = s["len"]
            entries.append(ee)
    clean = [e for e in entries if not e["leak"]["cluster_train"]
             and not (e["leak"]["exact"] and "train" in e["leak"]["exact"])]

    def boot_ci(x, n=2000, seed=0):
        if len(x) < 3:
            return None
        r = np.random.RandomState(seed)
        m = [r.choice(x, len(x), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    # (a) 序列先验
    def seq_prior(sub):
        dA = [e["logp_self"] - e["logp_randA_mean"] for e in sub]
        dB = [e["logp_self"] - e["logp_randB_mean"] for e in sub]
        return {
            "n": len(sub),
            "self_minus_rand_sameSS": {"mean": float(np.mean(dA)), "ci95": boot_ci(dA)},
            "self_minus_rand_ownSS": {"mean": float(np.mean(dB)), "ci95": boot_ci(dB)},
            "frac_self_beats_randA": float(np.mean(np.array(dA) > 0)),
            "per_nt": {"self": float(np.mean([e["logp_per_nt"] for e in sub])),
                       "randA": float(np.mean([e["logp_randA_mean"] / e["len"] for e in sub]))},
        }

    part_a = seq_prior(clean)
    part_a_sens = seq_prior(entries)   # 敏感性：含 40 条训练同簇条目

    # (b) 家族内 Kd 相关
    fam = {}
    for e in clean:
        if e["kd_nM"]:
            fam.setdefault(e["target_uid"], []).append(e)
    rhos, ns = [], []
    for uid, mem in fam.items():
        if len(mem) >= 3 and len({m["seq"] for m in mem}) >= 3:
            r = spearmanr([m["logp_self"] for m in mem],
                          [-np.log10(m["kd_nM"]) for m in mem]).statistic
            if np.isfinite(r):
                rhos.append(float(r)); ns.append(len(mem))
    part_b = {"n_families": len(rhos), "mean_rho": float(np.mean(rhos)) if rhos else None,
              "ci95": boot_ci(rhos), "median_rho": float(np.median(rhos)) if rhos else None,
              "frac_positive": float(np.mean(np.array(rhos) > 0)) if rhos else None,
              "family_sizes": ns, "n_families_ge2": sum(1 for v in fam.values() if len(v) >= 2)}

    # (c) 最优 Kd 排名检验（家族内最佳 Kd 的 logP 排名百分位；≥3 条 Kd 家族）
    pct = []
    for uid, mem in fam.items():
        if len(mem) >= 3 and len({m["seq"] for m in mem}) >= 3:
            best = min(mem, key=lambda m: m["kd_nM"])
            rank = 1 + sum(1 for m in mem if m["logp_self"] > best["logp_self"])
            pct.append((rank - 1) / (len(mem) - 1))
    part_c = {"n_families": len(pct),
              "mean_rank_pct_of_best_kd": float(np.mean(pct)) if pct else None,
              "ci95": boot_ci(pct),
              "p_one_sided_lt_half": float(wilcoxon(np.array(pct) - 0.5, alternative="less").pvalue)
              if len(pct) >= 10 else None}

    # (d) 全库相关 + 预算曲线（每家族内标准化后合并）
    zs_rho = []
    pooled = []
    for uid, mem in fam.items():
        if len(mem) >= 3 and len({m["seq"] for m in mem}) >= 3:
            lp = np.array([m["logp_self"] for m in mem])
            nk = -np.log10(np.array([m["kd_nM"] for m in mem]))
            if np.std(lp) > 0 and np.std(nk) > 0:
                pooled += list(zip((lp - lp.mean()) / lp.std(),
                                   (nk - nk.mean()) / nk.std()))
    if pooled:
        zl, zk = zip(*pooled)
        part_d = {"n_pooled": len(pooled),
                  "spearman_z": float(spearmanr(zl, zk).statistic),
                  "mean_z_logp_top_quartile_minus_bottom": None}
        arr = np.array(pooled)
        q = np.quantile(arr[:, 0], 0.75)
        part_d["mean_z_logp_top_quartile_minus_bottom"] = float(
            arr[arr[:, 0] >= q][:, 1].mean() - arr[arr[:, 0] < q][:, 1].mean())
    else:
        part_d = {"n_pooled": 0}

    # (e) top-50% 筛选灵敏度：只筛模型分数前一半时，捕获"家族内 Kd 前一半"的比例
    sens, base = [], []
    for uid, mem in fam.items():
        if len(mem) >= 4 and len({m["seq"] for m in mem}) >= 4:
            order = sorted(mem, key=lambda m: -m["logp_self"])
            top_half = order[:len(order) // 2]
            kd_sorted = sorted(mem, key=lambda m: m["kd_nM"])
            good = set(id(m) for m in kd_sorted[:len(kd_sorted) // 2])
            sens.append(sum(1 for m in top_half if id(m) in good) / len(good))
            base.append(0.5)
    part_e = {"n_families": len(sens),
              "sensitivity_at_50pct_budget": float(np.mean(sens)) if sens else None,
              "ci95": boot_ci(sens),
              "p_one_sided_gt_half": float(wilcoxon(np.array(sens) - 0.5, alternative="greater").pvalue)
              if len(sens) >= 10 else None}

    out = {"n_scored": len(entries), "n_clean": len(clean),
           "a_seq_prior": part_a, "a_seq_prior_sensitivity_all": part_a_sens,
           "b_family_kd_rho": part_b,
           "c_best_kd_rank": part_c, "d_pooled": part_d, "e_top_half_sensitivity": part_e}
    json.dump(out, open(REPORT_J, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['clean', 'leak', 'fold', 'score', 'report'])
    args = ap.parse_args()
    dict(clean=phase_clean, leak=phase_leak, fold=phase_fold,
         score=phase_score, report=phase_report)[args.phase]()


if __name__ == "__main__":
    main()
