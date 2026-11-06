#!/usr/bin/env python3
"""第 93 步：zero-shot 配体 Tanimoto 化学相似度分层（实验 8，纯 CPU）

问题：zs 93 链的配体与训练集配体的化学相似度是否解释 PACo 的相对优势？

流程：
  1. 链→配体：zs 93 链（splits_v2.json zero_shot_targets），stem→PDB（下划线前缀），
     PDB 的功能配体 = survey/candidates_ccd.json 该 PDB 的 ligands.other（多配体全保留）。
     用 survey/74_chain_ccd.json（115 界面链权威 chain→CCD 计数）对重叠链做一致性校验。
  2. 训练配体集合：data/s2_btrain/s2_samples_v2.pt 768 链（→ 510 unique PDB）的
     candidates_ccd other 配体并集。PDB 清单缓存于 survey/93_train_pdb_ids.json
     （由本脚本 --dump-train-pdbs 在 torch 环境生成）。
  3. SMILES：data/ccd/components.cif.gz（gemmi 解析，gzip 流式读入），
     _pdbx_chem_comp_descriptor 中优先 SMILES_CANONICAL(CACTVS)，否则任一 SMILES。
     RDKit MolFromSmiles → Morgan FP（radius=2, 2048 bit）→ Tanimoto。
     无 SMILES / RDKit 解析失败的配体剔除并记录。
  4. 每 zs 链：max_tanimoto = max over（自身配体 × 全部训练配体）。
     结合 survey/alayer_v2_retrained.json 的 per_chain（recovery_best / t01）：
     - 分箱 [0,0.2)/[0.2,0.4)/[0.4,0.6)/[0.6,1.0]：各方法均值 + PACo−RhoDesign Δ
       （逐链配对，Wilcoxon 双侧 + 单侧 greater）
     - 散点数据（x=max_tanimoto, y=Δrecovery_best）+ Spearman(相似度, Δ)

输出：survey/93_tanimoto_zs.json（临时文件 + rename）
环境：系统 python3（rdkit 2025.09 + gemmi 0.6.7 + scipy）；
     --dump-train-pdbs 需 torch 环境（/root/miniconda3/envs/mpl372_repro/bin/python）
"""
import os, sys, json, gzip
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
S2_PT = os.path.join(DATA_DIR, "s2_btrain", "s2_samples_v2.pt")
TRAIN_PDB_CACHE = os.path.join(SUR, "93_train_pdb_ids.json")
CCD_CIF = os.path.join(DATA_DIR, "ccd", "components.cif.gz")
OUT_TMP = os.path.join(SUR, "93_tanimoto_zs.json.tmp")
OUT = os.path.join(SUR, "93_tanimoto_zs.json")
METHODS = ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
BIN_LABELS = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,1.0]"]


def dump_train_pdbs():
    """torch 环境运行：s2 pt → PDB 清单缓存"""
    import torch
    d = torch.load(S2_PT, map_location="cpu", weights_only=False)
    pdbs = sorted({v["pdb_id"] for v in d.values()})
    tmp = TRAIN_PDB_CACHE + ".tmp"
    json.dump({"n_chains": len(d), "pdb_ids": pdbs}, open(tmp, "w"))
    os.replace(tmp, TRAIN_PDB_CACHE)
    print(f"训练链 {len(d)}，unique PDB {len(pdbs)} → {TRAIN_PDB_CACHE}")


def load_train_pdbs():
    if os.path.exists(TRAIN_PDB_CACHE):
        return json.load(open(TRAIN_PDB_CACHE))["pdb_ids"]
    import torch   # fallback：无缓存且当前环境有 torch
    d = torch.load(S2_PT, map_location="cpu", weights_only=False)
    return sorted({v["pdb_id"] for v in d.values()})


def load_smiles(needed):
    """components.cif.gz → {ccd: smiles}（只提取 needed 中的 CCD）"""
    import gemmi
    with gzip.open(CCD_CIF, "rt") as f:
        doc = gemmi.cif.read_string(f.read())
    out = {}
    for block in doc:
        cid = block.find_value("_chem_comp.id")
        if cid is None or cid.strip().strip("'\"") not in needed:
            continue
        cid = cid.strip().strip("'\"")
        tab = block.find("_pdbx_chem_comp_descriptor.",
                         ["type", "program", "descriptor"])
        canon, any_smi = None, None
        for row in tab:
            typ = str(row[0]).strip().strip("'\"")
            desc = str(row[2]).strip().strip("'\"")
            prog = str(row[1]).strip().strip("'\"")
            if "SMILES" not in typ.upper() or not desc:
                continue
            if typ.upper() == "SMILES_CANONICAL" and "CACTVS" in prog.upper():
                canon = desc
            if any_smi is None:
                any_smi = desc
        if canon or any_smi:
            out[cid] = canon or any_smi
    return out


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from scipy.stats import wilcoxon, spearmanr

    # ---- 1. 链→配体
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    zs_stems = list(sp["zero_shot_targets"])
    cands = json.load(open(os.path.join(SUR, "candidates_ccd.json")))
    pdb2lig = {c["pdb"]: list(dict.fromkeys(c["ligands"].get("other", [])))
               for c in cands}
    chain_ligs = {}
    no_lig = []
    for stem in zs_stems:
        pdb = stem.rsplit("_", 1)[0]
        ligs = pdb2lig.get(pdb, [])
        if ligs:
            chain_ligs[stem] = ligs
        else:
            no_lig.append(stem)
    print(f"zs 93 链：有功能配体 {len(chain_ligs)}，无 other 配体 {len(no_lig)} {no_lig}")

    # 74_chain_ccd 交叉校验（重叠链）
    cc74 = json.load(open(os.path.join(SUR, "74_chain_ccd.json")))
    overlap = sorted(set(cc74) & set(zs_stems))
    xcheck = {"n_overlap": len(overlap), "consistent": 0, "mismatch": {}}
    for stem in overlap:
        pdb = stem.rsplit("_", 1)[0]
        s74 = set(cc74[stem])
        scand = set(pdb2lig.get(pdb, []))
        if s74 == scand:
            xcheck["consistent"] += 1
        else:
            xcheck["mismatch"][stem] = {"74": sorted(s74), "candidates": sorted(scand)}
    print(f"74 交叉校验：重叠 {len(overlap)}，一致 {xcheck['consistent']}，"
          f"不一致 {len(xcheck['mismatch'])}")

    # ---- 2. 训练配体集
    train_pdbs = load_train_pdbs()
    train_ligs = set()
    n_train_pdb_hit = n_train_pdb_withlig = 0
    for p in train_pdbs:
        if p in pdb2lig:
            n_train_pdb_hit += 1
            if pdb2lig[p]:
                n_train_pdb_withlig += 1
                train_ligs.update(pdb2lig[p])
    train_ligs = sorted(train_ligs)
    print(f"训练 PDB {len(train_pdbs)}（candidates 命中 {n_train_pdb_hit}，"
          f"含 other 配体 {n_train_pdb_withlig}）→ 训练配体 {len(train_ligs)} 种")

    # ---- 3. SMILES + 指纹
    zs_ligs = sorted({l for v in chain_ligs.values() for l in v})
    needed = set(train_ligs) | set(zs_ligs)
    smi = load_smiles(needed)
    fps, no_smi, parse_fail = {}, [], []
    for ccd in sorted(needed):
        s = smi.get(ccd)
        if not s:
            no_smi.append(ccd)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            parse_fail.append(ccd)
            continue
        fps[ccd] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    train_fps = [fps[c] for c in train_ligs if c in fps]
    print(f"配体 {len(needed)} 种：无 SMILES {len(no_smi)}，RDKit 失败 {len(parse_fail)}，"
          f"有效 {len(fps)}；训练侧有效指纹 {len(train_fps)}")
    if not train_fps:
        raise RuntimeError("训练配体指纹为空")

    # ---- 4. 逐链 max_tanimoto
    per_chain = {}
    for stem in zs_stems:
        ligs = chain_ligs.get(stem, [])
        rec = {"pdb": stem.rsplit("_", 1)[0], "ligands": ligs,
               "ligands_with_fp": [l for l in ligs if l in fps]}
        if rec["ligands_with_fp"] and train_fps:
            per_lig = {}
            for l in rec["ligands_with_fp"]:
                sims = DataStructs.BulkTanimotoSimilarity(fps[l], train_fps)
                per_lig[l] = float(max(sims))
            rec["max_tanimoto_per_ligand"] = per_lig
            rec["max_tanimoto"] = max(per_lig.values())
        else:
            rec["max_tanimoto"] = None
        per_chain[stem] = rec

    # ---- 5. 结合 recovery 分层
    alayer = json.load(open(os.path.join(SUR, "alayer_v2_retrained.json")))["per_chain"]
    valid = [s for s in zs_stems
             if per_chain[s]["max_tanimoto"] is not None and s in alayer]
    print(f"进入分层：{len(valid)}/{len(zs_stems)} 链")
    for stem in valid:
        for m in METHODS:
            v = alayer[stem].get(m, {})
            per_chain[stem][f"recovery_best_{m}"] = v.get("recovery_best")
            per_chain[stem][f"recovery_t01_{m}"] = v.get("recovery_t01_mean")
        per_chain[stem]["delta_best_PACo_minus_RhoDesign"] = (
            alayer[stem]["PACoAptamer"]["recovery_best"]
            - alayer[stem]["RhoDesign"]["recovery_best"])
        per_chain[stem]["delta_t01_PACo_minus_RhoDesign"] = (
            alayer[stem]["PACoAptamer"]["recovery_t01_mean"]
            - alayer[stem]["RhoDesign"]["recovery_t01_mean"])

    def wilc(d):
        d = [x for x in d if x is not None]
        if len(d) < 10:
            return None, None
        try:
            return (float(wilcoxon(d, alternative="greater").pvalue),
                    float(wilcoxon(d).pvalue))
        except Exception:
            return None, None

    bin_rows = []
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        sel = [s for s in valid
               if lo <= per_chain[s]["max_tanimoto"] <= hi
               and (hi < 1.0 and per_chain[s]["max_tanimoto"] < hi or hi == 1.0)]
        row = {"bin": lab, "n": len(sel), "stems": sel}
        for m in METHODS:
            rb = [alayer[s][m]["recovery_best"] for s in sel
                  if m in alayer[s] and alayer[s][m].get("recovery_best") is not None]
            rt = [alayer[s][m]["recovery_t01_mean"] for s in sel
                  if m in alayer[s] and alayer[s][m].get("recovery_t01_mean") is not None]
            row[f"{m}_best_mean"] = float(np.mean(rb)) if rb else None
            row[f"{m}_t01_mean"] = float(np.mean(rt)) if rt else None
        d_best = [per_chain[s]["delta_best_PACo_minus_RhoDesign"] for s in sel]
        d_t01 = [per_chain[s]["delta_t01_PACo_minus_RhoDesign"] for s in sel]
        p1, p2 = wilc(d_best)
        row["delta_best_PACo_minus_RhoDesign"] = {
            "mean": float(np.mean(d_best)) if d_best else None,
            "median": float(np.median(d_best)) if d_best else None,
            "wilcoxon_p_one_sided_greater": p1, "wilcoxon_p_two_sided": p2}
        row["delta_t01_PACo_minus_RhoDesign"] = {
            "mean": float(np.mean(d_t01)) if d_t01 else None}
        bin_rows.append(row)
        print(f"  {lab}: n={len(sel)} " +
              " ".join(f"{m}={row[f'{m}_best_mean']:.3f}" for m in METHODS) +
              f" | Δ(T-R)={row['delta_best_PACo_minus_RhoDesign']['mean']:+.4f}"
              f" (p1={p1 if p1 is None else round(p1,4)})")

    # 散点 + Spearman
    x = [per_chain[s]["max_tanimoto"] for s in valid]
    y = [per_chain[s]["delta_best_PACo_minus_RhoDesign"] for s in valid]
    yt = [per_chain[s]["delta_t01_PACo_minus_RhoDesign"] for s in valid]
    sp_best = spearmanr(x, y)
    sp_t01 = spearmanr(x, yt)
    sp_methods = {}
    for m in METHODS:
        v = [alayer[s][m]["recovery_best"] for s in valid]
        r = spearmanr(x, v)
        sp_methods[m] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    scatter = [{"stem": s, "max_tanimoto": per_chain[s]["max_tanimoto"],
                "delta_best": per_chain[s]["delta_best_PACo_minus_RhoDesign"],
                "delta_t01": per_chain[s]["delta_t01_PACo_minus_RhoDesign"],
                "ligands": per_chain[s]["ligands"]} for s in valid]
    # overall（87 链全体，作对照行）
    overall = {"n": len(valid)}
    for m in METHODS:
        overall[f"{m}_best_mean"] = float(np.mean(
            [alayer[s][m]["recovery_best"] for s in valid]))
        overall[f"{m}_t01_mean"] = float(np.mean(
            [alayer[s][m]["recovery_t01_mean"] for s in valid]))
    p1o, p2o = wilc(y)
    overall["delta_best_PACo_minus_RhoDesign"] = {
        "mean": float(np.mean(y)), "median": float(np.median(y)),
        "wilcoxon_p_one_sided_greater": p1o, "wilcoxon_p_two_sided": p2o}
    overall["delta_t01_PACo_minus_RhoDesign"] = {"mean": float(np.mean(yt))}
    print(f"Spearman(sim, Δbest): rho={sp_best.statistic:.3f} p={sp_best.pvalue:.4g}")
    print(f"Spearman(sim, Δt01):  rho={sp_t01.statistic:.3f} p={sp_t01.pvalue:.4g}")
    for m in METHODS:
        r = sp_methods[m]
        print(f"Spearman(sim, best_{m}): rho={r['rho']:.3f} p={r['p']:.4g}")

    out = {
        "definition": {
            "zs_chains": len(zs_stems),
            "chain2lig": "candidates_ccd.json PDB 级 ligands.other（多配体取 max）",
            "train_ligands": f"s2_samples_v2.pt {len(train_pdbs)} unique PDB 的 "
                             "other 配体并集",
            "smiles": "components.cif.gz _pdbx_chem_comp_descriptor，优先 "
                      "SMILES_CANONICAL(CACTVS)，否则任一 SMILES",
            "fp": "RDKit Morgan radius=2 nBits=2048，Tanimoto（BulkTanimoto）",
            "recovery": "alayer_v2_retrained.json per_chain",
            "bins": BIN_LABELS,
        },
        "coverage": {
            "zs_no_ligand_stems": no_lig,
            "zs_n_with_ligand": len(chain_ligs),
            "train_pdb_n": len(train_pdbs),
            "train_pdb_hit_candidates": n_train_pdb_hit,
            "train_pdb_with_ligand": n_train_pdb_withlig,
            "train_ligand_ccd_n": len(train_ligs),
            "zs_ligand_ccd_n": len(zs_ligs),
            "ccd_no_smiles": no_smi,
            "ccd_rdkit_parse_fail": parse_fail,
            "train_ligands_with_fp": len(train_fps),
            "n_chains_in_stratification": len(valid),
        },
        "xcheck_74": xcheck,
        "overall": overall,
        "bins_table": bin_rows,
        "spearman": {
            "sim_vs_delta_best": {"rho": float(sp_best.statistic),
                                  "p": float(sp_best.pvalue)},
            "sim_vs_delta_t01": {"rho": float(sp_t01.statistic),
                                 "p": float(sp_t01.pvalue)},
            "sim_vs_recovery_best_per_method": sp_methods,
        },
        "scatter": scatter,
        "per_chain": per_chain,
    }
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, OUT)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    if "--dump-train-pdbs" in sys.argv:
        dump_train_pdbs()
    else:
        main()
