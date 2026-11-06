#!/usr/bin/env python3
"""第 95 步：LOFO 配体化学覆盖度实验（PACo stage-2 重训 + 评估）

阶段：
  map       链→CCD 归属：s2 768 链按 PDB 归属功能配体 CCD（candidates_ccd.json 的
            ligands.other 为候选；元素多重集整数倍 → 并集精确解 → ≥4 重原子贪心扩展
            (cov≥0.8) → 宽松 per-PDB 全候选），并用 74_chain_ccd.json（115 条界面链
            权威标注）以同管线在 rcsb_pdb 原始文件上重建归属做交叉校验
            → survey/95_chain_ccd.json
  families  CCD 级 + 家族级（flavins={FMN,FAD}, sam_sah={SAM,SAH}, ntp={ATP,GTP,CTP,UTP},
            aminoglycoside={PAR,GET,NMY,RIO,AM2,TOB}）双口径计数，定 top-5 家族
            → survey/95_families.json
  makedata  LOFO（剔除某家族全部链）与 only-k（仅 top-k 家族链）过滤集
            → data/s2_btrain/s2_samples_v2_lofo_{tag}.pt / s2_samples_v2_only{k}fam.pt
  train     逐变体调 tools/PACoAptamer/train.py（--joint --val_sel --epochs 8 --seed 43），
            已有 ckpt 跳过；日志 survey/95_train_{variant}.log
  eval      每变体在 zs 93 链（68 号口径：grnade_inputs_v2 特征，3 温度×100 采样，
            best=池内 max，t01=temp0.1 池均值）；LOFO 变体另做 115 界面链（74_iface_meta
            usable 且 0<n_iface≤50）界面/整体恢复率。池落 survey/95_pools/{variant}/
  collect   汇总 survey/95_lofo.json（家族表、训练 val 记录、zs/界面逐链与汇总、
            覆盖曲线、逐链 Δ vs 全量基线 + Wilcoxon p）

环境：/root/miniconda3/envs/rnasmol_service_gpu_env/bin/python（torch 2.1 + pyg 2.5）
GPU：CUDA_VISIBLE_DEVICES=1 python data/95_lofo.py --stage xxx
"""
import os, sys, json, gzip, argparse, collections, subprocess, time
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
sys.path.insert(0, os.path.join(ROOT, "tools"))
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
PY = "/root/miniconda3/envs/rnasmol_service_gpu_env/bin/python"
S2_SRC = os.path.join(DATA_DIR, "s2_btrain", "s2_samples_v2.pt")
CKPT_DIR = os.path.join(ROOT, "models", "PACoAptamer")
S1_INIT = "models/PACoAptamer/s1_batch4_seed43.pth"

FAMILIES = {   # 家族级合并（化学同族）；其余 CCD 自成一家
    "flavins": {"FMN", "FAD"},
    "sam_sah": {"SAM", "SAH"},
    "ntp": {"ATP", "GTP", "CTP", "UTP"},
    "aminoglycoside": {"PAR", "GET", "NMY", "RIO", "AM2", "TOB"},
}
CCD2FAM = {c: f for f, s in FAMILIES.items() for c in s}
TOP5_TAGS = ["aminoglycoside", "sam_sah", "cla", "ntp", "2ba"]   # families 阶段核实后使用
ONLY_KS = [1, 2, 3, 5]
BASELINE = "s2_v2_seed43_noiface"   # 既有主配置单模 ckpt（全量训练，不覆盖、只读评估）
FULL_CTL = "s2_v2_lofo_full_ctl"    # 全量复刻（run-to-run 噪声参照）

RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"


# ---------------------------------------------------------------- map
def stage_map():
    cands = json.load(open(os.path.join(SUR, "candidates_ccd.json")))
    per_pdb = {c["pdb"]: c["ligands"].get("other", []) for c in cands}
    glob = set()
    for c in cands:
        glob.update(c["ligands"].get("other", []))
    d = torch.load(S2_SRC, map_location="cpu", weights_only=False)
    need = set(glob)
    cc74 = json.load(open(os.path.join(SUR, "74_chain_ccd.json")))
    for v in cc74.values():
        need.update(v.keys())
    ccd_elem = parse_ccd_elements(need)

    # s2 PDB 级归属
    bypdb = {}
    for k, v in d.items():
        r = bypdb.setdefault(v["pdb_id"], {"chains": [], "M": collections.Counter()})
        r["chains"].append(k)
        if not r["M"]:
            r["M"] = collections.Counter(e.upper() for e in v["ligand_elements"])
    attr = {}
    for p, rec in bypdb.items():
        attr[p] = attribute(rec["M"], per_pdb.get(p, []), glob, ccd_elem)
        attr[p]["chains"] = sorted(rec["chains"])
        attr[p]["n_lig_atoms"] = sum(rec["M"].values())

    # 交叉校验：74_chain_ccd 的 115 链（test 界面链，与 s2 不相交）用同管线在
    # rcsb_pdb 原始 PDB 上重建 M → 归属 → 与权威 CCD 标注比对（交集即一致）
    pdb_dir = os.path.join(DATA_DIR, "rcsb_pdb")
    xval = {"agree": 0, "disagree": 0, "detail": {}}
    for ch, theirs in cc74.items():
        p = ch.rsplit("_", 1)[0]
        M = pdb_multiset(os.path.join(pdb_dir, f"{p}.pdb"), glob)
        if M is None:
            xval["detail"][ch] = {"theirs": sorted(theirs), "mine": None}
            continue
        a = attribute(M, per_pdb.get(p, []), glob, ccd_elem)
        mine = set(a["ccds"])
        ok = bool(mine & set(theirs))
        xval["agree" if ok else "disagree"] += 1
        if not ok:
            xval["detail"][ch] = {"theirs": sorted(theirs), "mine": sorted(mine),
                                  "method": a["method"]}
    method_stats = collections.Counter(a["method"] for a in attr.values())
    out = {"note": "s2 768 链→CCD 归属（PDB 级）；method: element_exact/element_union/"
                   "greedy_pdb(per-PDB候选内贪心,cov≥0.8)/greedy_ext(扩展到全局白名单)/"
                   "loose_pdb（宽松=per-PDB 全候选）；贪心仅计 ≥4 重原子 CCD",
           "method_stats_pdb": dict(method_stats),
           "n_pdb": len(attr), "n_chain": sum(len(a["chains"]) for a in attr.values()),
           "crossval_74": xval, "attr": attr}
    json.dump(out, open(os.path.join(SUR, "95_chain_ccd.json"), "w"), ensure_ascii=False)
    print("map 完成:", dict(method_stats),
          f"| 74 交叉校验 agree {xval['agree']} / {xval['agree']+xval['disagree']}")


def parse_ccd_elements(need):
    """components.cif.gz → {CCD: Counter(重原子元素)}"""
    found = collections.defaultdict(collections.Counter)
    state, cols = "idle", []
    with gzip.open(os.path.join(DATA_DIR, "ccd", "components.cif.gz"), "rt",
                   encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if state == "idle":
                if s == "loop_":
                    state, cols = "maybe_cols", []
            elif state == "maybe_cols":
                if s.startswith("_chem_comp_atom."):
                    cols.append(s)
                elif s.startswith("_"):
                    state = "skip_loop"
                elif s == "loop_" or s == "#":
                    state = "idle" if s == "#" else "maybe_cols"
                else:
                    idx_comp = cols.index("_chem_comp_atom.comp_id")
                    idx_sym = cols.index("_chem_comp_atom.type_symbol")
                    state = "in_rows"
            elif state == "skip_loop":
                if s == "#":
                    state = "idle"
                elif s == "loop_":
                    state, cols = "maybe_cols", []
            if state == "in_rows":
                if s == "#" or s == "loop_" or s.startswith("_"):
                    state = "idle" if s != "loop_" else "maybe_cols"
                    if s == "loop_":
                        cols = []
                    continue
                parts = s.split()
                if len(parts) < len(cols):
                    continue
                comp = parts[idx_comp]
                if comp in need:
                    sym = parts[idx_sym].strip().upper()
                    if sym != "H":
                        found[comp][sym] += 1
    return {k: collections.Counter(v) for k, v in found.items()}


def is_multiple(M, E):
    ks = []
    for e, n in E.items():
        if M.get(e, 0) == 0 or M[e] % n:
            return None
        ks.append(M[e] // n)
    if not ks or len(set(ks)) != 1:
        return None
    k = ks[0]
    return k if set(M) == set(E) else None


def solve_union(M, cands, ccd_elem, max_k=40):
    cands = [c for c in cands if c in ccd_elem]
    best = [None]
    def dfs(i, rem, acc):
        if best[0] is not None:
            return
        if i == len(cands):
            if all(v == 0 for v in rem.values()):
                best[0] = dict(acc)
            return
        c = cands[i]; Ec = ccd_elem[c]
        ub = min([rem.get(e, 0) // n for e, n in Ec.items()] + [max_k])
        for k in range(ub, -1, -1):
            rem2 = dict(rem); ok = True
            for e, n in Ec.items():
                rem2[e] = rem2.get(e, 0) - k * n
                if rem2[e] < 0:
                    ok = False; break
            if not ok:
                continue
            acc[c] = k
            dfs(i + 1, rem2, acc)
            del acc[c]
    dfs(0, dict(M), {})
    return best[0]


def greedy_decompose(M, pool, ccd_elem, min_atoms=4):
    """M 的贪心覆盖分解。CCD 组成先投影到 M 实有元素上（容忍旧 PDB 部分原子未沉积，
    如 1ILX 的 CLA 只沉积了 C/N/MG），kmax 按投影组成取整；仅计 ≥min_atoms 的 CCD。"""
    rem = collections.Counter(M); hit = {}
    while True:
        best_c, best_k, best_atoms = None, 0, 0
        for c in pool:
            E = ccd_elem.get(c)
            if not E:
                continue
            Ep = {e: n for e, n in E.items() if M.get(e, 0) > 0}
            if sum(Ep.values()) < min_atoms:
                continue
            kmax = min(rem.get(e, 0) // n for e, n in Ep.items())
            if kmax >= 1:
                atoms = kmax * sum(Ep.values())
                if atoms > best_atoms:
                    best_c, best_k, best_atoms = c, kmax, atoms
        if best_c is None:
            break
        hit[best_c] = hit.get(best_c, 0) + best_k
        Ep = {e: n for e, n in ccd_elem[best_c].items() if M.get(e, 0) > 0}
        for e, n in Ep.items():
            rem[e] = max(0, rem[e] - best_k * n)
        if sum(rem.values()) == 0:
            break
    return hit, sum(rem.values())


def attribute(M, cands_pdb, glob, ccd_elem):
    """PDB 配体元素多重集 → CCD 归属"""
    singles = []
    for c in cands_pdb:
        if c in ccd_elem:
            k = is_multiple(M, ccd_elem[c])
            if k:
                singles.append((c, k))
    if len(singles) == 1:
        return {"ccds": {singles[0][0]: singles[0][1]}, "method": "element_exact"}
    if len(singles) > 1:
        return {"ccds": dict(singles), "method": "element_exact_multi"}
    u = solve_union(M, cands_pdb, ccd_elem)
    if u:
        return {"ccds": {c: k for c, k in u.items() if k > 0}, "method": "element_union"}
    # 先在 per-PDB 候选内贪心（防全局白名单里同分异构 CCD 抢走归属，如 1ILX CLA→POH 假象）
    total = sum(M.values())
    hit, left = greedy_decompose(M, sorted(cands_pdb), ccd_elem)
    if total and (total - left) / total >= 0.8 and hit:
        return {"ccds": hit, "method": "greedy_pdb",
                "greedy_cov": round((total - left) / total, 3)}
    pool = sorted(set(cands_pdb) | glob)
    hit, left = greedy_decompose(M, pool, ccd_elem)
    if total and (total - left) / total >= 0.8 and hit:
        return {"ccds": hit, "method": "greedy_ext",
                "greedy_cov": round((total - left) / total, 3)}
    return {"ccds": {c: None for c in cands_pdb}, "method": "loose_pdb"}


def pdb_multiset(path, glob):
    if not os.path.exists(path):
        return None
    M = collections.Counter()
    for line in open(path):
        if line[:6].strip() != "HETATM" or line[16] not in (" ", "A"):
            continue
        if line[17:20].strip() not in glob:
            continue
        M[(line[76:78].strip() or "C").upper()] += 1
    return M


def chain_families(attr):
    """链 → 家族集合（CCD 经 CCD2FAM 映射）"""
    out = {}
    for p, a in attr.items():
        fams = {CCD2FAM.get(c, c.lower()) for c in a["ccds"]}
        for ch in a["chains"]:
            out[ch] = sorted(fams)
    return out


# ---------------------------------------------------------------- families
def stage_families():
    m = json.load(open(os.path.join(SUR, "95_chain_ccd.json")))
    attr = m["attr"]
    ccd_cnt = collections.Counter()
    fam_cnt = collections.Counter()
    fam_pdbs = collections.defaultdict(set)
    for p, a in attr.items():
        nch = len(a["chains"])
        fams = {CCD2FAM.get(c, c.lower()) for c in a["ccds"]}
        for c in a["ccds"]:
            ccd_cnt[c] += nch
        for f in fams:
            fam_cnt[f] += nch
            fam_pdbs[f].add(p)
    names = json.load(open("/tmp/ccd_names.json")) if os.path.exists("/tmp/ccd_names.json") else {}
    table_ccd = [{"ccd": c, "chains": n, "name": names.get(c, "")}
                 for c, n in ccd_cnt.most_common()]
    table_fam = [{"family": f, "chains": n, "n_pdb": len(fam_pdbs[f])}
                 for f, n in fam_cnt.most_common()]
    top5 = [t["family"] for t in table_fam[:5]]
    out = {"families_def": {k: sorted(v) for k, v in FAMILIES.items()},
           "ccd_level": table_ccd, "family_level": table_fam, "top5": top5}
    json.dump(out, open(os.path.join(SUR, "95_families.json"), "w"),
              indent=1, ensure_ascii=False)
    print("CCD top10:", [(t["ccd"], t["chains"]) for t in table_ccd[:10]])
    print("家族 top10:", [(t["family"], t["chains"]) for t in table_fam[:10]])
    print("TOP5:", top5)


# ---------------------------------------------------------------- makedata
def stage_makedata():
    m = json.load(open(os.path.join(SUR, "95_chain_ccd.json")))
    fams_of_chain = chain_families(m["attr"])
    fam_table = json.load(open(os.path.join(SUR, "95_families.json")))
    top5 = fam_table["top5"]
    assert top5 == TOP5_TAGS, f"top5 与脚本常量不一致: {top5} vs {TOP5_TAGS}"
    d = torch.load(S2_SRC, map_location="cpu", weights_only=False)
    manifest = {}
    # LOFO：剔除该家族全部链
    for tag in TOP5_TAGS:
        keep = {k: v for k, v in d.items() if tag not in fams_of_chain.get(k, [])}
        path = os.path.join(DATA_DIR, "s2_btrain", f"s2_samples_v2_lofo_{tag}.pt")
        if not os.path.exists(path):
            torch.save(keep, path)
        manifest[f"lofo_{tag}"] = {"file": os.path.basename(path), "n_chains": len(keep),
                                   "removed": len(d) - len(keep)}
        print(f"lofo_{tag}: keep {len(keep)} (removed {len(d)-len(keep)})")
    # only-k：仅保留 top-k 家族并集链
    for k in ONLY_KS:
        ks = set(top5[:k])
        keep = {kk: v for kk, v in d.items() if set(fams_of_chain.get(kk, [])) & ks}
        path = os.path.join(DATA_DIR, "s2_btrain", f"s2_samples_v2_only{k}fam.pt")
        if not os.path.exists(path):
            torch.save(keep, path)
        manifest[f"only{k}fam"] = {"file": os.path.basename(path), "n_chains": len(keep),
                                   "kept_families": sorted(ks)}
        print(f"only{k}fam: keep {len(keep)}")
    # full_ctl：全量复刻（同一文件即可，--s2_data s2_samples_v2.pt 名会命中旧缓存——
    # 用拷贝名 s2_samples_v2_fullctl.pt 走独立缓存）
    path = os.path.join(DATA_DIR, "s2_btrain", "s2_samples_v2_fullctl.pt")
    if not os.path.exists(path):
        torch.save(d, path)
    manifest["full_ctl"] = {"file": os.path.basename(path), "n_chains": len(d)}
    json.dump(manifest, open(os.path.join(SUR, "95_datasets.json"), "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- train
VARIANTS = [f"lofo_{t}" for t in TOP5_TAGS] + [f"only{k}fam" for k in ONLY_KS] + ["full_ctl"]


def variant_s2data(v):
    if v == "full_ctl":
        return "s2_samples_v2_fullctl.pt"
    if v.startswith("lofo_"):
        return f"s2_samples_v2_{v}.pt"
    return f"s2_samples_v2_{v}.pt"   # only{k}fam


def stage_train():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    for v in VARIANTS:
        ckpt = os.path.join(CKPT_DIR, f"s2_v2_{v}.pth")
        if os.path.exists(ckpt):
            print(f"[skip] {v} ckpt 已存在")
            continue
        log = os.path.join(SUR, f"95_train_{v}.log")
        cmd = [PY, "tools/PACoAptamer/train.py", "--stage", "s2", "--init", S1_INIT,
               "--splits_v2", "--s2_data", variant_s2data(v), "--joint", "--val_sel",
               "--epochs", "8", "--seed", "43", "--out", f"s2_v2_{v}"]
        print(f"[train] {v} ...", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f"  rc={r.returncode} {time.time()-t0:.0f}s → {ckpt}", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"训练失败 {v}，见 {log}")


# ---------------------------------------------------------------- eval
def get_m38_m68():
    import importlib.util
    spec38 = importlib.util.spec_from_file_location("m38", os.path.join(DATA_DIR, "38_eval_paco_btest.py"))
    m38 = importlib.util.module_from_spec(spec38)
    spec38.loader.exec_module(m38)
    spec68 = importlib.util.spec_from_file_location("m68", os.path.join(DATA_DIR, "68_eval_paco_av2.py"))
    m68 = importlib.util.module_from_spec(spec68)
    spec68.loader.exec_module(m68)
    return m38, m68


def zs_stems():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    return sorted(sp["zero_shot_targets"])


def iface_stems():
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    return sorted(k for k, v in meta.items()
                  if v and v.get("usable") and 0 < v["n_iface"] <= 50), meta


def eval_variant(m38, m68, variant, ckpt, layer):
    """layer ∈ {zs, iface}；池落 survey/95_pools/{variant}/zs|iface/{stem}.json
    zs：68 号 A 层口径（grnade_inputs_v2 特征）；iface：74 号 B 层口径（paco_s1_v2
    缓存特征 + 74_iface_meta 界面掩码）——115 链在 A 层只有 29 条覆盖，必须走 B 层。"""
    pool_dir = os.path.join(SUR, "95_pools", variant, layer)
    os.makedirs(pool_dir, exist_ok=True)
    if layer == "zs":
        stems, meta, s1data = zs_stems(), None, None
    else:
        stems, meta = iface_stems()
        from PACoAptamer.data_utils import load_s1
        s1data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    device = torch.device("cuda:0")
    model, alphabet = m38.load_model(ckpt, device)
    n_s = m38.N_SAMPLES
    results = {}
    for i, stem in enumerate(stems):
        pf = os.path.join(pool_dir, f"{stem}.json")
        if os.path.exists(pf):
            results[stem] = {k: v for k, v in json.load(open(pf)).items()
                             if k not in ("native", "pool")}
            continue
        if layer == "zs":
            d, status = m68.build_feature(stem)
            if d is None:
                print(f"  [SKIP] {stem}: {status}", flush=True)
                continue
        else:
            if stem not in s1data:
                print(f"  [SKIP] {stem}: not_in_s1_cache", flush=True)
                continue
            d = s1data[stem]
        pool = []
        for t in m38.TEMPS:
            pool.extend(m38.sample_chain(model, alphabet, d, device, t, n_s, False))
        native = d["seq"]
        match = [[a == b for a, b in zip(s, native)] for s in pool]
        recs = [sum(m_) / len(native) for m_ in match]
        rec = {"len": len(native), "recovery_best": max(recs),
               "recovery_t01_mean": float(np.mean(recs[:n_s])), "n_pool": len(pool)}
        if layer == "iface":
            im = np.array(meta[stem]["interface"], dtype=bool)
            if im.sum() > 0 and len(im) == len(native):
                irecs = [float(np.mean(np.array(m_)[im])) for m_ in match]
                rec["iface_best"] = max(irecs)
                rec["iface_t01_mean"] = float(np.mean(irecs[:n_s]))
                rec["n_iface"] = int(im.sum())
        results[stem] = rec
        json.dump({"native": native, "pool": pool, **rec}, open(pf, "w"))
    del model
    torch.cuda.empty_cache()
    agg = {"n": len(results),
           "best_mean": float(np.mean([r["recovery_best"] for r in results.values()])),
           "t01_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()]))}
    if layer == "iface":
        sub = [r for r in results.values() if "iface_best" in r]
        agg["n_iface_ok"] = len(sub)
        if sub:
            agg["iface_best_mean"] = float(np.mean([r["iface_best"] for r in sub]))
            agg["iface_t01_mean"] = float(np.mean([r["iface_t01_mean"] for r in sub]))
    return {"summary": agg, "per_chain": results}


def stage_eval():
    m38, m68 = get_m38_m68()
    variants = [(BASELINE, os.path.join(CKPT_DIR, BASELINE + ".pth"))]
    variants += [(f"s2_v2_{v}", os.path.join(CKPT_DIR, f"s2_v2_{v}.pth")) for v in VARIANTS]
    out_path = os.path.join(SUR, "95_eval.json")
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name, ckpt in variants:
        if not os.path.exists(ckpt):
            print(f"[eval-skip] {name}: 无 ckpt")
            continue
        for layer in ("zs", "iface"):
            # 时间预算口径：iface 层只做 LOFO 5 变体 + 基线 + full_ctl
            if layer == "iface" and not (name == BASELINE or "full_ctl" in name or "lofo_" in name):
                continue
            key = f"{name}|{layer}"
            if key in done:
                print(f"[skip] {key}")
                continue
            print(f"[eval] {key}", flush=True)
            t0 = time.time()
            done[key] = eval_variant(m38, m68, name, ckpt, layer)
            done[key]["wall_s"] = round(time.time() - t0, 1)
            json.dump(done, open(out_path, "w"), ensure_ascii=False)
            print(f"  → {done[key]['summary']}", flush=True)
    json.dump(done, open(out_path, "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- collect
def stage_collect():
    from scipy.stats import wilcoxon
    fam = json.load(open(os.path.join(SUR, "95_families.json")))
    dsm = json.load(open(os.path.join(SUR, "95_datasets.json")))
    ev = json.load(open(os.path.join(SUR, "95_eval.json")))
    # 训练 val 记录
    train_rec = {}
    for v in VARIANTS:
        log = os.path.join(SUR, f"95_train_{v}.log")
        if not os.path.exists(log):
            continue
        vals = []
        for line in open(log):
            line = line.strip()
            if line.startswith("S2 Epoch") and "VAL" in line:
                ep = int(line.split()[2].rstrip(":"))
                val = float(line.split("VAL")[1].split("|")[0].strip())
                vals.append({"epoch": ep, "val": val})
        best = max((x["val"] for x in vals), default=None)
        train_rec[v] = {"epoch_val": vals, "best_val": best}
    base_zs = ev[f"{BASELINE}|zs"]["per_chain"]
    def delta_vs_base(per):
        common = sorted(set(per) & set(base_zs))
        d = [per[s]["recovery_best"] - base_zs[s]["recovery_best"] for s in common]
        dt = [per[s]["recovery_t01_mean"] - base_zs[s]["recovery_t01_mean"] for s in common]
        p = wilcoxon(d).pvalue if len(d) >= 10 and any(abs(x) > 1e-12 for x in d) else None
        return {"n": len(common), "delta_best_mean": float(np.mean(d)),
                "delta_t01_mean": float(np.mean(dt)),
                "wilcoxon_p_best": float(p) if p is not None else None,
                "per_chain_delta": {s: round(per[s]["recovery_best"] - base_zs[s]["recovery_best"], 4)
                                    for s in common}}
    zs_table = {}
    for key, rec in ev.items():
        name, layer = key.split("|")
        if layer != "zs":
            continue
        row = {"summary": rec["summary"]}
        if name != BASELINE:
            row["vs_baseline"] = delta_vs_base(rec["per_chain"])
        zs_table[name] = row
    iface_table = {k.split("|")[0]: v["summary"] for k, v in ev.items() if k.endswith("|iface")}
    # 覆盖曲线
    cov = []
    for k in ONLY_KS:
        name = f"s2_v2_only{k}fam"
        if name in zs_table:
            cov.append({"k": k, "n_train_chains": dsm[f"only{k}fam"]["n_chains"],
                        "zs_best": zs_table[name]["summary"]["best_mean"],
                        "zs_t01": zs_table[name]["summary"]["t01_mean"]})
    full = {"n_train_chains": dsm["full_ctl"]["n_chains"],
            "zs_best": zs_table.get(BASELINE, {}).get("summary", {}).get("best_mean"),
            "zs_t01": zs_table.get(BASELINE, {}).get("summary", {}).get("t01_mean")}
    out = {"experiment": "LOFO 配体化学覆盖度（PACo s2 重训，seed43 joint+val_sel 8ep）",
           "families": fam, "datasets": dsm, "train_val": train_rec,
           "baseline_ckpt": BASELINE, "zs_table": zs_table, "iface_table": iface_table,
           "coverage_curve": cov, "full_reference": full}
    json.dump(out, open(os.path.join(SUR, "95_lofo.json"), "w"), indent=1, ensure_ascii=False)
    print("collect 完成 → survey/95_lofo.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["map", "families", "makedata", "train", "eval", "collect"])
    args = ap.parse_args()
    {"map": stage_map, "families": stage_families, "makedata": stage_makedata,
     "train": stage_train, "eval": stage_eval, "collect": stage_collect}[args.stage]()
