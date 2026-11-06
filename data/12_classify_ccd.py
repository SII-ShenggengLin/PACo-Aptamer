#!/usr/bin/env python3
"""RNA-配体复合物摸底（第 3 阶段：CCD 权威类型重分类）

用 PDB CCD 字典（components.cif 的 _chem_comp.type）对 candidates.json 中
"other" 类配体做权威分类：
- RNA/DNA linking / 末端变体 → 修饰核苷酸（非配体，补全 RNA 残基表）
- peptide 类 → 肽/氨基酸
- saccharide → 糖
- non-polymer → 一般配体（功能候选）
配合金属离子表、水表、常见结晶剂表，输出修正后的功能配体池。
"""
import json, os
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT_DIR = os.path.join(DATA_DIR, "survey")

METAL_IONS = {"NA","MG","K","CA","MN","ZN","FE","FE2","CL","BR","I","CO","NI","CU",
              "HG","CD","PT","AU","AG","LI","RB","CS","SR","BA","AL","GA","IN","TL",
              "PB","BI","SE","OS","RU","RH","PD","IR","MO","W","V","CR","AS","SB",
              "TE","PO","GD","TB","DY","ER","YB","SM","EU","HO","LA","CE","PR","ND",
              "PM","LU","SC","TI","ZR","HF","NB","TA","RE","TC","GE","SN","HGO",
              "CU1","ZN2","MN3","FE1","FE3","MG2","CA2","KO4","IOD",
              # 非金属离子/含氧阴离子（CCD 也是 NON-POLYMER）
              "NH4","SO4","PO4","CLO","BEF","ALF","SE4","MOO","VO4","WO4","TMA",
              "F","CL","BR","IOD","CN","NCO","OCN","SCN","NO2","NO3","CO2","CO3",
              "ACY","ACE","ACT","FORM","FMT","BCT","OXL","TAR","CIT","SUC","FUM",
              "MAL","MLA","GAL","GLU","TLA","IPA","PG4","MPD","PEG","PGE","1PE",
              "2PE","3PE","4PE","5PE","6PE","7PE","8PE","PE3","PE4","PE5","PE6",
              "PE7","PE8","PEO","POP","PTF","P33","P35","P3G","P3K","PG6","PG8",
              # 金属复合物/无机离子（CCD 亦为 NON-POLYMER）
              "OHX","IRI","AF3","NH2","NI3","CO3","NCO","CN","OCN","SCN","NO2",
              "NO3","CO2","CYN","N3","NCP"}
WATER = {"HOH","WAT","DOD","H2O"}
CRYSTAL_COMMON = {"GOL","EDO","DMS","MPD","TMAO","CDL","CHL","DGA","DGD","DSL",
                  "DTV","EDA","EOH","EPC","LMT","LDA","D12","C8E","CXS","SPD",
                  "SPM","PUT","CAD","IMD","DMI","U1","GTT","A2G","SUC","FUM",
                  "MAL","ACY","ACE","BCR","PHM","UNX","DUM","UNL","BOG","OCT",
                  "DPG","PO3","TRS","BME","DTT","DTD","TCE","EPE","HEP","MES",
                  "TAR","SIN","MAE","MOH","MLI","MO3","MO4","MPD1","TMT","FRU",
                  "BGC","FUC","MAN","NAG","BMA","SIA","GLC","GAL","MES1","HEP1",
                  "MSE","SF4","FES","3FE","CLF","SF3","SFO","DOC","CHT","HTO",
                  # 结晶剂/去垢剂/溶剂（CCD 查证）
                  "IPH","1N7","S9L","CHAPS","CHAPSO","BOG","DDAO","LDAO","Z3G",
                  "C14","C15","C16","C17","C18","FLC","TLA","POV","B3P","EPQ"}

def norm_type(t):
    if not t:
        return None
    t = t.upper().replace('"', '').strip()
    return t

def ccd_category(t):
    """CCD type → 大类：rna_mod / dna_mod / peptide / saccharide / polymer_other / non_polymer"""
    t = norm_type(t)
    if t is None:
        return None
    if "RNA" in t:
        return "rna_mod"
    if "DNA" in t:
        return "dna_mod"
    if "PEPTIDE" in t:
        return "peptide"
    if "SACCHARIDE" in t:
        return "saccharide"
    if "POLYMER" in t or "LINKING" in t:
        return "polymer_other"
    if "NON-POLYMER" in t or "ION" in t:
        return "non_polymer"
    return "unknown"

def main():
    ccd = json.load(open(os.path.join(DATA_DIR, "ccd", "ccd_types.json")))
    cands = json.load(open(os.path.join(OUT_DIR, "candidates.json")))

    # CCD 类型查表（大小写归一）
    type_lookup = {k.upper(): v["type"] for k, v in ccd.items()}
    name_lookup = {k.upper(): v["name"] for k, v in ccd.items()}

    # 每个 HET 名 → 最终类别
    final_cat = {}
    stats = defaultdict(int)
    # 修饰核苷酸全集（CCD RNA linking，供补全 RNA 残基表）
    rna_mod_full = {rn for rn, t in type_lookup.items() if ccd_category(t) == "rna_mod"}

    for entry in cands:
        other_ligs = entry["ligands"].get("other", [])
        new_ligs = {"water": entry["ligands"].get("water", []),
                    "ion": entry["ligands"].get("ion", []),
                    "mod_nuc": list(set(entry["ligands"].get("mod_nuc", []))),  # 10 号脚本表
                    "crystal": entry["ligands"].get("crystal", []),
                    "other": []}   # 待重分类
        for rn in other_ligs:
            rn_u = rn.upper()
            if rn_u in final_cat:
                cat = final_cat[rn_u]
            else:
                t = type_lookup.get(rn_u)
                cc = ccd_category(t)
                if rn_u in WATER:
                    cat = "water"
                elif rn_u in METAL_IONS:
                    cat = "ion"
                elif rn_u in CRYSTAL_COMMON:
                    cat = "crystal"
                elif cc == "rna_mod":
                    cat = "mod_nuc"          # 修饰核苷酸（权威）
                elif cc == "dna_mod":
                    cat = "mod_nuc"
                elif cc == "peptide":
                    cat = "peptide"
                elif cc == "saccharide":
                    cat = "crystal"          # 糖类：结晶添加剂为主
                elif cc in ("non_polymer", "unknown", "polymer_other"):
                    cat = "other"            # 功能配体候选
                else:
                    cat = "other"
                final_cat[rn_u] = cat
            if cat in ("water", "ion", "mod_nuc", "crystal", "peptide", "other"):
                new_ligs.setdefault(cat, [])
                if rn not in new_ligs[cat]:
                    new_ligs[cat].append(rn)
            stats[cat] += 1
        entry["ligands"] = new_ligs
        entry["ligands_ccd"] = {rn: (type_lookup.get(rn.upper()), name_lookup.get(rn.upper()))
                                for rn in other_ligs}

    # 修正后统计
    def has(entry, cat):
        return len(entry["ligands"].get(cat, [])) > 0

    n_rna_lig = sum(1 for r in cands if has(r, "other"))
    n_rna_lig_plus_prot = sum(1 for r in cands if has(r, "other") and r["n_prot"] > 0)
    n_rna_lig_only = sum(1 for r in cands if has(r, "other") and r["n_prot"] == 0)
    n_rna_crystal_only = sum(1 for r in cands if not has(r, "other") and has(r, "crystal"))
    n_rna_peptide = sum(1 for r in cands if has(r, "peptide"))

    # 功能配体频次（修正后）
    lig_freq = defaultdict(int)
    for r in cands:
        for rn in r["ligands"].get("other", []):
            lig_freq[rn] += 1

    summary = {
        "ccd_total": len(ccd),
        "rna_mod_ccd_full": len(rna_mod_full),
        "n_rna_functional_ligand": n_rna_lig,
        "n_rna_functional_ligand_only_rna": n_rna_lig_only,
        "n_rna_functional_ligand_plus_protein": n_rna_lig_plus_prot,
        "n_rna_crystal_only_after_ccd": n_rna_crystal_only,
        "n_rna_peptide_ligand": n_rna_peptide,
        "functional_ligand_freq": dict(sorted(lig_freq.items(), key=lambda x: -x[1])),
        "het_category_stats": dict(stats),
        "rna_mod_full_list": sorted(rna_mod_full),
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "ccd_summary.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(cands, open(os.path.join(OUT_DIR, "candidates_ccd.json"), "w"), indent=1, ensure_ascii=False)

    print(f"CCD 条目: {len(ccd)}  RNA 修饰核苷酸(CCD 权威): {len(rna_mod_full)}", flush=True)
    print(f"HET 重分类统计: {dict(stats)}", flush=True)
    print(f"修正后：RNA+功能配体条目 {n_rna_lig}  (纯RNA {n_rna_lig_only} | RNA+蛋白 {n_rna_lig_plus_prot})", flush=True)
    print(f"RNA+仅肽类配体: {n_rna_peptide} | RNA+仅结晶剂(修正后): {n_rna_crystal_only}", flush=True)
    print(f"\n修正后 Top 40 功能配体:", flush=True)
    for rn, c in list(summary["functional_ligand_freq"].items())[:40]:
        name = name_lookup.get(rn.upper(), "")
        print(f"  {rn:6s} {c:5d}  {name[:60]}", flush=True)

if __name__ == "__main__":
    main()
