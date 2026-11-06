#!/usr/bin/env python3
"""RNA-配体复合物数据集摸底（第 1 阶段：纯文本统计，分钟级）

目标：从本地 24,252 个 PDB 统计含 RNA 链 + 配体（HETATM 非水非离子）的条目，
区分 RNA-小分子 / RNA-蛋白结合态复合物，评估"靶标条件化"方向数据可行性。

坑位处理（对齐 04 文档防泄漏清单）：
- HETATM 三段分类：水 / 金属离子 / 修饰核苷酸（RNA 残基表必须完整，清单第 10 条）
  → 输出"HETATM 残基名 × 链类型"交叉表，事后补全修饰表
- 链 ID 大小写敏感（05 文档坑位）
- 分辨率 REMARK 2 用 line[8:11].strip()=="2"（05 文档坑位）
- release_date 从 BGSU 索引（/tmp/bgsu_pdbs.json）关联（时间切分基础）
"""
import os, re, json, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
OUT_DIR = os.path.join(DATA_DIR, "survey")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 残基/配体分类表 ----
# RNA 标准 + 常见修饰核苷酸（HETATM 中出现的不算配体）
RNA_RES = {"A", "C", "G", "U", "I",
           "1MA", "5MC", "5MU", "PSU", "2MG", "7MG", "M2G", "OMG", "OMC", "OMU",
           "MIA", "YYG", "H2U", "5BU", "20U", "UR3", "T6A", "BGH", "G7M", "QUO",
           "1MG", "2OM", "5NC", "6MA", "AMP", "CMP", "GMP", "UMP", "IPP", "GDP",
           # 扩充：常见 RNA 修饰（PDB CCD）
           "4SU", "5OHU", "5MC2", "5MU1", "6IA", "7MC", "8OG", "A23", "A2M", "ACPC",
           "AD2", "ADP", "AGM", "AMD", "AMO", "APT", "ARG", "B24", "BND", "BRU",
           "C2L", "C43", "C5L", "C7M", "CBR", "CCC", "CCL", "CD0", "CFL", "CG1",
           "CH6", "CMO", "CNO", "CRB", "CR2", "CSL", "CTP", "DB2", "DBI", "DDG",
           "DHE", "DHU", "DUM", "G25", "G2L", "G36", "G37", "G49", "G7M", "GDO",
           "GDP", "GH3", "GHT", "GIA", "GLC", "GLN", "GMU", "GNG", "GNP", "GOM",
           "GPN", "GSO", "GTP", "GTP2", "H2U", "H3U", "HBI", "HDU", "HIA", "HIS",
           "HMU", "HOH", "HPU", "HTA", "I4U", "IC", "IG", "IMC", "IU", "KAG",
           "KAN", "LHG", "LYG", "M2A", "M2G", "M3U", "M5C", "M5U", "M7A", "M7G",
           "MA6", "MAG", "MAM", "MAN", "MAP", "MAQ", "MAU", "MBG", "MDO", "MED",
           "MG3", "MGU", "MIA", "MIS", "MNU", "MO6", "MOG", "MOW", "MPG", "MPN",
           "MTU", "MVA", "N2G", "N6G", "NC6", "NMG", "NMP", "NRQ", "O2G", "O3G",
           "O4U", "O5U", "O8G", "OA1", "OMU", "ONH", "ONL", "ONP", "ORD", "OTG",
           "P5U", "PBG", "PGP", "PGV", "PHU", "PMP", "PPN", "PR5", "PSD", "PTR",
           "PU", "PYO", "Q", "QUO", "R5P", "RC", "RG", "RU", "S2C", "S2G", "S4U",
           "S6G", "SEL", "SGP", "SHP", "SMP", "SRA", "SRS", "SST", "STU", "SUL",
           "T6A", "TAF", "TBR", "TCY", "TDR", "TFA", "TGP", "TGT", "THX", "TLC",
           "TMD", "TMS", "TRC", "TRP", "TRQ", "TSP", "TTS", "TUR", "TYP", "U25",
           "U2L", "U31", "U36", "U8O", "UAR", "UBD", "UCL", "UDP", "UFR", "UFT",
           "UMA", "UMC", "UML", "UMO", "UMP", "UNK", "UOX", "UPA", "UPL", "UPT",
           "UR3", "URD", "URM", "URO", "US1", "US5", "USE", "USM", "UTR", "VAS",
           "WBU", "XCP", "XUA", "YCM", "YG", "YOF", "YYG", "ZAD", "ZCT", "ZDU",
           "ZGU", "ZTH", "ZYD", "ZYT"}
# 氨基酸（ATOM 记录）——注意 ARG/HIS/LYS/GLY 等与配体残基名冲突，配体判定用链类型区分
AA_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
          "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
          "SEC","PYL","MSE"}
DNA_RES = {"DA", "DC", "DG", "DT", "DI"}
# 金属离子（HETATM）
METAL_IONS = {"NA","MG","K","CA","MN","ZN","FE","FE2","CL","BR","I","CO","NI","CU",
              "HG","CD","PT","AU","AG","LI","RB","CS","SR","BA","AL","GA","IN","TL",
              "PB","BI","SE","OS","RU","RH","PD","IR","MO","W","V","CR","AS","SB",
              "TE","PO","GD","TB","DY","ER","YB","SM","EU","HO","LA","CE","PR","ND",
              "PM","LU","SC","TI","ZR","HF","NB","TA","RE","TC","GE","SN","HGO",
              "CU1","ZN2","MN3","FE1","FE3","MG2","CA2","KO4","IOD"}
WATER = {"HOH","WAT","DOD","H2O"}
# 常见结晶试剂/缓冲液/沉淀剂（HETATM 中常见，通常不是功能配体；摸底时单独标记）
CRYSTAL_COMMON = {"GOL","EDO","PEG","PGE","PGR","MPD","DMS","SO4","PO4","ACT","FMT",
                  "TRS","BME","DTT","DTD","TCE","CIT","MSE","EPE","HEP","MES",
                  "P4G","P6G","1PE","2PE","3PE","4PE","5PE","6PE","7PE","8PE",
                  "P33","P35","P3G","P3K","PG4","PG6","PG8","P6G","PE3","PE4",
                  "PE5","PE6","PE7","PE8","PEO","POP","PTF","PGT","MES1","HEP1",
                  "TAR","SIN","MAE","MOH","MLI","MO3","MO4","MPD1","TMAO","TMT",
                  "CDL","CHL","DGA","DGD","DOD","DSL","DTV","EDA","EOH","EPC",
                  "GLC","FUC","MAN","NAG","BMA","GAL","SIA","LMT","LDA","D12",
                  "C8E","CXS","SPD","SPM","PUT","CAD","IMD","DMI","DMS","U1",
                  "GTT","A2G","AMP","CIT","SUC","FUM","MAL","ACY","ACE","BCR",
                  "PHM","UNX","DUM","UNL","BOG","OCT","DPG","PE8","PO3"}

def parse_pdb(path):
    """返回 (resolution, atom_chains, het_dict)
    atom_chains: {chain: {resname}}  ATOM 记录链残基
    het_dict: {chain: {resname}}     HETATM 记录（水/离子也记录，分类时区分）
    """
    resolution = None
    atom_chains = defaultdict(set)
    het_chains = defaultdict(set)
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "REMARK" and line[8:11].strip() == "2":
                m = re.search(r"RESOLUTION\.\s+([\d.]+)", line)
                if m:
                    resolution = float(m.group(1))
            elif rec == "ATOM":
                chain = line[21]
                resname = line[17:20].strip()
                atom_chains[chain].add(resname)
            elif rec == "HETATM":
                chain = line[21]
                resname = line[17:20].strip()
                het_chains[chain].add(resname)
    return resolution, dict(atom_chains), dict(het_chains)

def classify_chain(atom_res, het_res):
    """判定链类型。修饰核苷酸若在 HETATM 且属于 RNA_RES → 计入 RNA 链。"""
    all_res = set(atom_res) | {r for r in het_res if r in RNA_RES}
    has_rna = any(r in RNA_RES for r in all_res)
    has_dna = any(r in DNA_RES for r in all_res)
    has_aa  = any(r in AA_RES for r in all_res)
    if has_aa:
        return "mixed" if (has_rna or has_dna) else "protein"
    if has_rna and has_dna:
        return "mixed_nuc"
    if has_rna:
        return "rna"
    if has_dna:
        return "dna"
    return "other"

def classify_ligand(resname):
    """配体三段分类: water / ion / mod_nuc / crystal / other"""
    if resname in WATER:
        return "water"
    if resname in METAL_IONS:
        return "ion"
    if resname in RNA_RES:
        return "mod_nuc"
    if resname in CRYSTAL_COMMON:
        return "crystal"
    return "other"

def scan_one(fname):
    """扫描单个 PDB，返回条目级摘要"""
    pdb_id = fname[:-4].upper()
    path = os.path.join(PDB_DIR, fname)
    try:
        resolution, atom_chains, het_chains = parse_pdb(path)
    except Exception as e:
        return {"pdb": pdb_id, "error": str(e)}

    # 链类型判定
    chain_types = {}
    for ch in set(atom_chains) | set(het_chains):
        chain_types[ch] = classify_chain(atom_chains.get(ch, set()), het_chains.get(ch, set()))

    n_rna = sum(1 for t in chain_types.values() if t == "rna")
    n_prot = sum(1 for t in chain_types.values() if t == "protein")
    n_mixed = sum(1 for t in chain_types.values() if t == "mixed")

    # 配体收集（条目级去重 + 分类计数）
    ligands = defaultdict(set)   # 分类 -> {resname}
    ligand_chains = {}           # resname -> [链ID]
    het_cross = []               # (resname, 链类型) 交叉表
    for ch, resnames in het_chains.items():
        ctype = chain_types.get(ch, "other")
        for rn in resnames:
            cat = classify_ligand(rn)
            ligands[cat].add(rn)
            ligand_chains.setdefault(rn, set()).add(ch)
            het_cross.append((rn, ctype))

    entry = {
        "pdb": pdb_id,
        "resolution": resolution,
        "n_rna": n_rna, "n_prot": n_prot, "n_mixed": n_mixed,
        "chain_types": chain_types,
        "ligands": {k: sorted(v) for k, v in ligands.items()},
        "ligand_chains": {k: sorted(v) for k, v in ligand_chains.items()},
    }
    return entry

def main():
    files = sorted(f for f in os.listdir(PDB_DIR) if f.endswith(".pdb"))
    print(f"待扫描: {len(files)} 个 PDB", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(ex.map(scan_one, files, chunksize=64)):
            results.append(r)
            if (i + 1) % 5000 == 0:
                print(f"进度: {i+1}/{len(files)}", flush=True)

    # ---- 聚合统计 ----
    n_parse_error = sum(1 for r in results if "error" in r)
    ok = [r for r in results if "error" not in r]

    def has_cat(entry, cat):
        return len(entry["ligands"].get(cat, [])) > 0

    n_any_rna = sum(1 for r in ok if r["n_rna"] > 0)
    n_rna_only = sum(1 for r in ok if r["n_rna"] > 0 and r["n_prot"] == 0 and r["n_mixed"] == 0)
    n_rna_prot = sum(1 for r in ok if r["n_rna"] > 0 and (r["n_prot"] > 0 or r["n_mixed"] > 0))
    n_rna_lig = sum(1 for r in ok if r["n_rna"] > 0 and has_cat(r, "other"))
    n_rna_lig_only_crystal = sum(1 for r in ok if r["n_rna"] > 0 and not has_cat(r, "other") and has_cat(r, "crystal"))
    n_rna_ion_only = sum(1 for r in ok if r["n_rna"] > 0 and not has_cat(r, "other") and not has_cat(r, "crystal") and has_cat(r, "ion"))
    n_rna_lig_prot = sum(1 for r in ok if r["n_rna"] > 0 and r["n_prot"] > 0 and has_cat(r, "other"))

    # 配体频次（"other" 类，条目级）
    lig_freq = defaultdict(int)
    lig_rna_contact_count = defaultdict(int)  # 该配体所在条目中 RNA 链数>0 的计数
    for r in ok:
        for rn in r["ligands"].get("other", []):
            lig_freq[rn] += 1
            if r["n_rna"] > 0:
                lig_rna_contact_count[rn] += 1

    # 修饰核苷酸全集（供补全 RNA 残基表）
    mod_freq = defaultdict(int)
    for r in ok:
        for rn in r["ligands"].get("mod_nuc", []):
            mod_freq[rn] += 1

    # HETATM × 链类型交叉表（找漏判的修饰核苷酸）
    cross = defaultdict(int)
    for r in ok:
        for ch, resnames in r["ligand_chains"].items():
            pass  # 交叉表在 scan 阶段收集，这里重新遍历条目
    # 简化：重新从 results 构建
    cross2 = defaultdict(int)
    for r in ok:
        for rn, chains in r["ligand_chains"].items():
            for ch in chains:
                cross2[(rn, r["chain_types"].get(ch, "other"))] += 1

    # 时间分布（BGSU release_date）
    bgsu = {}
    if os.path.exists("/tmp/bgsu_pdbs.json"):
        bgsu = json.load(open("/tmp/bgsu_pdbs.json"))
    def date_of(pdb_id):
        v = bgsu.get(pdb_id.lower())
        return v.get("release_date", None) if v else None

    time_dist = defaultdict(int)
    for r in ok:
        d = date_of(r["pdb"])
        if d:
            time_dist[d[:4]] += 1

    # 含配体的条目（候选池，供第 2 阶段空间分析）
    candidates = [r for r in ok if r["n_rna"] > 0 and has_cat(r, "other")]
    for r in candidates:
        r["release_date"] = date_of(r["pdb"])

    summary = {
        "total_pdb": len(files),
        "parse_ok": len(ok),
        "parse_error": n_parse_error,
        "n_any_rna": n_any_rna,
        "n_rna_only": n_rna_only,
        "n_rna_prot": n_rna_prot,
        "n_rna_ligand_other": n_rna_lig,
        "n_rna_crystal_only": n_rna_lig_only_crystal,
        "n_rna_ion_only": n_rna_ion_only,
        "n_rna_ligand_and_protein": n_rna_lig_prot,
        "ligand_freq": dict(sorted(lig_freq.items(), key=lambda x: -x[1])),
        "ligand_freq_in_rna_entries": dict(sorted(lig_rna_contact_count.items(), key=lambda x: -x[1])),
        "mod_nuc_freq": dict(sorted(mod_freq.items(), key=lambda x: -x[1])),
        "het_resname_x_chaintype": {f"{k[0]}|{k[1]}": v for k, v in sorted(cross2.items(), key=lambda x: -x[1])},
        "release_year_dist": dict(sorted(time_dist.items())),
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(candidates, open(os.path.join(OUT_DIR, "candidates.json"), "w"), indent=1, ensure_ascii=False)

    print("\n===== 摸底统计（纯文本，第 1 阶段） =====", flush=True)
    print(f"总条目: {summary['total_pdb']}  解析失败: {n_parse_error}", flush=True)
    print(f"含 RNA 链条目: {n_any_rna}  (纯 RNA: {n_rna_only} | RNA+蛋白: {n_rna_prot})", flush=True)
    print(f"RNA + 功能配体(other): {n_rna_lig}", flush=True)
    print(f"  ├ RNA + 功能配体 + 蛋白: {n_rna_lig_prot}", flush=True)
    print(f"RNA + 仅结晶试剂: {n_rna_lig_only_crystal}", flush=True)
    print(f"RNA + 仅离子: {n_rna_ion_only}", flush=True)
    print(f"\nTop 30 功能配体（全部条目）: ", flush=True)
    for rn, c in list(summary["ligand_freq"].items())[:30]:
        print(f"  {rn:6s} {c}", flush=True)
    print(f"\nTop 30 功能配体（仅 RNA 条目）: ", flush=True)
    for rn, c in list(summary["ligand_freq_in_rna_entries"].items())[:30]:
        print(f"  {rn:6s} {c}", flush=True)
    print(f"\n时间分布(年): {dict(summary['release_year_dist'])}", flush=True)
    print(f"\n结果写入: {OUT_DIR}/summary.json, candidates.json", flush=True)

if __name__ == "__main__":
    main()
