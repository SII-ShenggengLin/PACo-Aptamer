#!/usr/bin/env python3
"""第 106 步：Tanimoto 化学新颖性结论的指纹稳健性（补充实验 S2，纯 CPU）

93_tanimoto_zs.json 用 RDKit Morgan(radius=2, 2048bit) 指纹算出：87 条 zs 链配体对
247 种训练配体的 max Tanimoto 与 PACo−RhoDesign 优势反相关
（Δt01 ρ=−0.401 p=1.2e-4；Δbest ρ=−0.249 p=0.020）。
本步对同一 87 链 × 同一训练配体库，换用两种指纹重算，检验结论是否依赖指纹选择：
  (a) MACCS keys（166 个预定义结构子结构位）
  (b) RDKit topological fingerprint（RDKFingerprint 默认参数：路径长 1–7，2048 bit）
并把 Morgan 口径完整复算一遍作为内部校验（应逐链重现 93 号 JSON 的 max_tanimoto）。

配体清单 / SMILES 提取 / 链→配体映射逐字复用 data/93_tanimoto_zs.py 的逻辑
（train PDB 清单缓存 93_train_pdb_ids.json；SMILES 来自 components.cif.gz
_pdbx_chem_comp_descriptor，优先 SMILES_CANONICAL(CACTVS)）。
recovery 与 Δ 取自 survey/alayer_v2_retrained.json per_chain（与 93 号一致，
并断言与 93 存储值相等）。

输出：survey/106_tanimoto_robust.json（临时文件 + rename）
环境：系统 python3（rdkit 2025.09 + gemmi 0.6.7 + scipy，同 93 步）
"""
import gzip
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
TRAIN_PDB_CACHE = os.path.join(SUR, "93_train_pdb_ids.json")
CCD_CIF = os.path.join(DATA_DIR, "ccd", "components.cif.gz")
T93 = os.path.join(SUR, "93_tanimoto_zs.json")
OUT_TMP = os.path.join(SUR, "106_tanimoto_robust.json.tmp")
OUT = os.path.join(SUR, "106_tanimoto_robust.json")
METHODS = ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
BIN_LABELS = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,1.0]"]
FP_NAMES = ["morgan", "maccs", "rdkit_topo"]


def load_smiles(needed):
    """components.cif.gz → {ccd: smiles}（逐字复用 93_tanimoto_zs.py）"""
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


def wilc(d):
    """同 93_tanimoto_zs.py：n<10 不做 Wilcoxon"""
    d = [x for x in d if x is not None]
    if len(d) < 10:
        return None, None
    try:
        return (float(wilcoxon(d, alternative="greater").pvalue),
                float(wilcoxon(d).pvalue))
    except Exception:
        return None, None


def in_bin(x, lo, hi):
    """同 93：前三个箱左闭右开，最后 [0.6,1.0] 双闭"""
    return lo <= x <= hi and (hi < 1.0 and x < hi or hi == 1.0)


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, MACCSkeys, RDKFingerprint

    t93 = json.load(open(T93))

    # ---- 1. 链→配体（复刻 93：splits_v2 zero_shot_targets + candidates_ccd other）
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    zs_stems = list(sp["zero_shot_targets"])
    cands = json.load(open(os.path.join(SUR, "candidates_ccd.json")))
    pdb2lig = {c["pdb"]: list(dict.fromkeys(c["ligands"].get("other", [])))
               for c in cands}
    chain_ligs = {}
    for stem in zs_stems:
        ligs = pdb2lig.get(stem.rsplit("_", 1)[0], [])
        if ligs:
            chain_ligs[stem] = ligs
    assert set(zs_stems) == set(t93["per_chain"]), "zs 链清单与 93 不一致"
    for stem, rec in t93["per_chain"].items():
        assert rec["ligands"] == chain_ligs.get(stem, []), f"{stem} 配体映射与 93 不一致"

    # ---- 2. 训练配体集（复刻 93：510 unique PDB 的 other 配体并集）
    train_pdbs = json.load(open(TRAIN_PDB_CACHE))["pdb_ids"]
    train_ligs = set()
    for p in train_pdbs:
        train_ligs.update(pdb2lig.get(p, []))
    train_ligs = sorted(train_ligs)
    assert len(train_ligs) == t93["coverage"]["train_ligand_ccd_n"] == 247

    # ---- 3. SMILES → mol → 三种指纹
    zs_ligs = sorted({l for v in chain_ligs.values() for l in v})
    needed = set(train_ligs) | set(zs_ligs)
    smi = load_smiles(needed)
    mols, no_smi, parse_fail = {}, [], []
    for ccd in sorted(needed):
        s = smi.get(ccd)
        if not s:
            no_smi.append(ccd)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            parse_fail.append(ccd)
            continue
        mols[ccd] = mol
    assert sorted(no_smi) == sorted(t93["coverage"]["ccd_no_smiles"])
    assert sorted(parse_fail) == sorted(t93["coverage"]["ccd_rdkit_parse_fail"])

    fps = {"morgan": {}, "maccs": {}, "rdkit_topo": {}}
    for ccd, mol in mols.items():
        fps["morgan"][ccd] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps["maccs"][ccd] = MACCSkeys.GenMACCSKeys(mol)
        fps["rdkit_topo"][ccd] = RDKFingerprint(mol)  # 默认 minPath=1 maxPath=7 2048bit
    train_fps = {fp: [fps[fp][c] for c in train_ligs if c in fps[fp]]
                 for fp in FP_NAMES}
    n_train_fp = {fp: len(v) for fp, v in train_fps.items()}
    assert n_train_fp["morgan"] == t93["coverage"]["train_ligands_with_fp"] == 242
    print(f"配体 {len(needed)} 种，mol 有效 {len(mols)}；"
          f"训练侧指纹 {n_train_fp}")

    # ---- 4. 逐链 max_tanimoto（三种指纹）
    sim = {fp: {} for fp in FP_NAMES}  # fp -> stem -> max_tanimoto or None
    per_lig = {fp: {} for fp in FP_NAMES}
    for stem in zs_stems:
        ligs = [l for l in chain_ligs.get(stem, []) if l in mols]
        for fp in FP_NAMES:
            if ligs:
                pl = {l: float(max(DataStructs.BulkTanimotoSimilarity(
                    fps[fp][l], train_fps[fp]))) for l in ligs}
                per_lig[fp][stem] = pl
                sim[fp][stem] = max(pl.values())
            else:
                per_lig[fp][stem] = {}
                sim[fp][stem] = None
    valid = [s for s in zs_stems if sim["morgan"][s] is not None]
    assert len(valid) == t93["coverage"]["n_chains_in_stratification"] == 87
    # 指纹不改变命运：mol 相同 ⇒ 各指纹有效链集合相同
    for fp in FP_NAMES:
        assert [s for s in zs_stems if sim[fp][s] is not None] == valid

    # ---- 5. recovery/Δ（alayer_v2_retrained，断言与 93 一致）
    alayer = json.load(open(os.path.join(SUR, "alayer_v2_retrained.json")))["per_chain"]
    d_best, d_t01 = {}, {}
    for s in valid:
        d_best[s] = (alayer[s]["PACoAptamer"]["recovery_best"]
                     - alayer[s]["RhoDesign"]["recovery_best"])
        d_t01[s] = (alayer[s]["PACoAptamer"]["recovery_t01_mean"]
                    - alayer[s]["RhoDesign"]["recovery_t01_mean"])
        r93 = t93["per_chain"][s]
        assert abs(d_best[s] - r93["delta_best_PACo_minus_RhoDesign"]) < 1e-12
        assert abs(d_t01[s] - r93["delta_t01_PACo_minus_RhoDesign"]) < 1e-12

    # ---- 6. Morgan 复算校验（应逐链重现 93 的 max_tanimoto）
    morgan_diffs = [abs(sim["morgan"][s] - t93["per_chain"][s]["max_tanimoto"])
                    for s in valid]
    x_m = [sim["morgan"][s] for s in valid]
    y_b = [d_best[s] for s in valid]
    y_t = [d_t01[s] for s in valid]
    morgan_recheck = {
        "max_abs_diff_max_tanimoto": float(max(morgan_diffs)),
        "spearman_sim_vs_delta_best": {
            "rho": float(spearmanr(x_m, y_b).statistic),
            "p": float(spearmanr(x_m, y_b).pvalue),
            "reference": t93["spearman"]["sim_vs_delta_best"]},
        "spearman_sim_vs_delta_t01": {
            "rho": float(spearmanr(x_m, y_t).statistic),
            "p": float(spearmanr(x_m, y_t).pvalue),
            "reference": t93["spearman"]["sim_vs_delta_t01"]},
    }
    print(f"Morgan 复算：max|Δsim|={max(morgan_diffs):.3g} "
          f"ρ_best={morgan_recheck['spearman_sim_vs_delta_best']['rho']:.4f} "
          f"ρ_t01={morgan_recheck['spearman_sim_vs_delta_t01']['rho']:.4f}")

    # ---- 7. 各指纹 Spearman + MACCS 分箱表
    def spearman_block(fp):
        x = [sim[fp][s] for s in valid]
        return {"sim_vs_delta_best": {"rho": float(spearmanr(x, y_b).statistic),
                                      "p": float(spearmanr(x, y_b).pvalue)},
                "sim_vs_delta_t01": {"rho": float(spearmanr(x, y_t).statistic),
                                     "p": float(spearmanr(x, y_t).pvalue)}}

    spearman_all = {fp: spearman_block(fp) for fp in FP_NAMES}

    def bin_table(fp):
        rows = []
        for (lo, hi), lab in zip(BINS, BIN_LABELS):
            sel = [s for s in valid if in_bin(sim[fp][s], lo, hi)]
            row = {"bin": lab, "n": len(sel), "stems": sel}
            for m in METHODS:
                row[f"{m}_best_mean"] = float(np.mean(
                    [alayer[s][m]["recovery_best"] for s in sel])) if sel else None
                row[f"{m}_t01_mean"] = float(np.mean(
                    [alayer[s][m]["recovery_t01_mean"] for s in sel])) if sel else None
            db = [d_best[s] for s in sel]
            dt = [d_t01[s] for s in sel]
            p1, p2 = wilc(db)
            row["delta_best_PACo_minus_RhoDesign"] = {
                "mean": float(np.mean(db)) if db else None,
                "median": float(np.median(db)) if db else None,
                "wilcoxon_p_one_sided_greater": p1,
                "wilcoxon_p_two_sided": p2}
            row["delta_t01_PACo_minus_RhoDesign"] = {
                "mean": float(np.mean(dt)) if dt else None}
            rows.append(row)
        return rows

    bins_all = {fp: bin_table(fp) for fp in FP_NAMES}
    for fp in FP_NAMES:
        r = spearman_all[fp]
        print(f"[{fp}] ρ_best={r['sim_vs_delta_best']['rho']:+.4f} "
              f"(p={r['sim_vs_delta_best']['p']:.4g}) "
              f"ρ_t01={r['sim_vs_delta_t01']['rho']:+.4f} "
              f"(p={r['sim_vs_delta_t01']['p']:.4g})")
        for row in bins_all[fp]:
            dm = row["delta_best_PACo_minus_RhoDesign"]["mean"]
            print(f"    {row['bin']}: n={row['n']:2d} "
                  f"Δbest={dm if dm is None else round(dm, 4)} "
                  f"p2={row['delta_best_PACo_minus_RhoDesign']['wilcoxon_p_two_sided']}")

    # ---- 8. 指纹间一致性（87 链 max-sim 的秩相关）
    consistency = {}
    for a, b in [("morgan", "maccs"), ("morgan", "rdkit_topo"),
                 ("maccs", "rdkit_topo")]:
        xa = [sim[a][s] for s in valid]
        xb = [sim[b][s] for s in valid]
        r = spearmanr(xa, xb)
        consistency[f"{a}_vs_{b}"] = {
            "spearman_rho": float(r.statistic), "p": float(r.pvalue),
            "pearson_r": float(np.corrcoef(xa, xb)[0, 1]),
            "median_abs_diff": float(np.median(
                [abs(x1 - x2) for x1, x2 in zip(xa, xb)]))}
        print(f"consistency {a} vs {b}: ρ={r.statistic:.4f} (p={r.pvalue:.3g})")

    # ---- 9. 逐链明细 + 输出
    per_chain = {}
    for s in valid:
        rec = t93["per_chain"][s]
        per_chain[s] = {
            "pdb": rec["pdb"], "ligands": rec["ligands"],
            "ligands_with_fp": rec["ligands_with_fp"],
            "max_tanimoto_morgan": sim["morgan"][s],
            "max_tanimoto_maccs": sim["maccs"][s],
            "max_tanimoto_rdkit_topo": sim["rdkit_topo"][s],
            "max_tanimoto_per_ligand_maccs": per_lig["maccs"][s],
            "max_tanimoto_per_ligand_rdkit_topo": per_lig["rdkit_topo"][s],
            "delta_best_PACo_minus_RhoDesign": d_best[s],
            "delta_t01_PACo_minus_RhoDesign": d_t01[s],
            "recovery_best_PACoAptamer": alayer[s]["PACoAptamer"]["recovery_best"],
            "recovery_t01_PACoAptamer": alayer[s]["PACoAptamer"]["recovery_t01_mean"],
            "recovery_best_RhoDesign": alayer[s]["RhoDesign"]["recovery_best"],
            "recovery_t01_RhoDesign": alayer[s]["RhoDesign"]["recovery_t01_mean"],
        }

    out = {
        "meta": {
            "step": 106,
            "question": "93 号结论（PACo−RhoDesign 优势与训练配体化学相似度反相关）"
                        "是否依赖指纹选择？",
            "base_experiment": "93_tanimoto_zs.json（Morgan radius=2, 2048bit）",
            "fingerprints": {
                "morgan": "RDKit MorganFingerprint radius=2 nBits=2048（复算校验）",
                "maccs": "RDKit MACCSkeys（166 结构键 + bit0，167 bit）",
                "rdkit_topo": "RDKit RDKFingerprint 默认参数"
                              "（minPath=1, maxPath=7, fpSize=2048, 拓扑路径指纹）",
            },
            "chain_ligand_map": "同 93：splits_v2.json zero_shot_targets 93 链 × "
                                "candidates_ccd.json PDB 级 ligands.other；"
                                "已断言逐链配体与 93 存储一致",
            "train_ligands": "93_train_pdb_ids.json 510 unique PDB 的 other 配体并集 "
                             "= 247 种（断言 == 93 coverage）",
            "smiles": "components.cif.gz _pdbx_chem_comp_descriptor，优先 "
                      "SMILES_CANONICAL(CACTVS)，否则任一 SMILES（load_smiles "
                      "逐字复用 93 脚本）",
            "valid_chains": "87 = 93 − 5 无 other 配体 − 1（6PQ7_C，配体 OXV 无 "
                            "SMILES）；三种指纹的 mol 相同，有效链集合不变",
            "deltas": "alayer_v2_retrained.json per_chain，PACo−RhoDesign；"
                      "已断言与 93 存储 Δ 一致（<1e-12）",
            "stats": "Spearman ρ/p（双侧）；分箱同 93：[0,0.2)/[0.2,0.4)/[0.4,0.6)/"
                     "[0.6,1.0]，Wilcoxon 双侧 + 单侧 greater（n<10 不做）",
            "date": "2026-10-16",
        },
        "coverage": {
            "n_zs": len(zs_stems), "n_valid": len(valid),
            "train_ligand_ccd_n": len(train_ligs),
            "train_ligands_with_fp": n_train_fp,
            "ccd_no_smiles": no_smi, "ccd_rdkit_parse_fail": parse_fail,
        },
        "morgan_recheck": morgan_recheck,
        "spearman": spearman_all,
        "bins_table_maccs": bins_all["maccs"],
        "bins_table_rdkit_topo": bins_all["rdkit_topo"],
        "bins_table_morgan_recheck": bins_all["morgan"],
        "fingerprint_consistency": consistency,
        "per_chain": per_chain,
    }
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, OUT)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
