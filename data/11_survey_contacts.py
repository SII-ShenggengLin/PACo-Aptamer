#!/usr/bin/env python3
"""RNA-配体复合物数据集摸底（第 2 阶段：空间接触分析）

输入：survey/candidates.json（含 RNA 链 + "other" 配体的条目）
对每个条目计算：
- 每个功能配体残基 → RNA 链原子最小距离（判定配体是否真正接触 RNA）
- 每个功能配体残基 → 蛋白链原子最小距离
- 蛋白链 → RNA 链最小距离（判定 RNA-蛋白结合态）
界面定义：最小重原子距离 < 5.0Å（与 NA-MPNN interface mask 一致的逻辑）

输出：survey/contacts.json + 汇总统计
"""
import os, re, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
OUT_DIR = os.path.join(DATA_DIR, "survey")
CONTACT_CUTOFF = 5.0  # Å

# 复用 10 号脚本的残基表
RNA_RES = {"A", "C", "G", "U", "I",
           "1MA", "5MC", "5MU", "PSU", "2MG", "7MG", "M2G", "OMG", "OMC", "OMU",
           "MIA", "YYG", "H2U", "5BU", "20U", "UR3", "T6A", "BGH", "G7M", "QUO",
           "1MG", "2OM", "5NC", "6MA", "AMP", "CMP", "GMP", "UMP", "IPP", "GDP",
           "4SU", "5OHU", "5MC2", "5MU1", "6IA", "7MC", "8OG", "A23", "A2M", "ACPC",
           "AD2", "ADP", "AGM", "AMD", "AMO", "APT", "B24", "BND", "BRU",
           "C2L", "C43", "C5L", "C7M", "CBR", "CCC", "CCL", "CD0", "CFL", "CG1",
           "CH6", "CMO", "CNO", "CRB", "CR2", "CSL", "CTP", "DB2", "DBI", "DDG",
           "DHE", "DHU", "DUM", "G25", "G2L", "G36", "G37", "G49", "G7M", "GDO",
           "GDP", "GH3", "GHT", "GIA", "GLC", "GLN", "GMU", "GNG", "GNP", "GOM",
           "GPN", "GSO", "GTP", "GTP2", "H2U", "H3U", "HBI", "HDU", "HIA", "HIS",
           "HMU", "HPU", "HTA", "I4U", "IC", "IG", "IMC", "IU", "KAG",
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
AA_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
          "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
          "SEC","PYL","MSE"}
DNA_RES = {"DA", "DC", "DG", "DT", "DI"}

# 已知"适配体/核糖开关类功能配体"的 HET 名单（人工 curated，用于命中高价值靶标）
KNOWN_TARGET_LIGANDS = {
    # 核糖开关配体
    "SAM","SAM1","SAM2","SAM3","SAM8","SF4",           # SAM I-IV 等
    "FMN","FMN1","FMN2",                               # FMN 核糖开关
    "TPP","TPP1","TPH","TPP2",                         # TPP 核糖开关
    "THF","THG","FOL","FOL1","THA",                    # THF/叶酸
    "GTP","GDP","ATP","ADP","CTP","UTP","GMP","AMP",   # 核苷酸
    "C2G","CDG","CG2","CMP",                           # c-di-GMP
    "2DG","CDA",                                       # c-di-AMP
    "3GP","3GU",                                       # cGMP
    "GUN","ADE","XAN","HXU","HYP","URC",               # 碱基
    "PQ1","PQO","7AQ","PQ2",                           # preQ1
    "G6P","G6M","G6P1","G6S",                          # GlcN6P
    "MCD","MOC","MOX",                                 # MoCo
    "ZTP","ZMP",                                       # ZMP/ZTP
    "LYS","ARG","GLY","CIT","DAP","HIS",               # 氨基酸（注意与蛋白残基区分）
    "SMZ","SMA",                                       # SAM/SAH 变体
    "SAH",                                             # S-腺苷同型半胱氨酸
    "AAG","AP5","AP6",                                 # 其他
    # 经典适配体靶标
    "THE",                                             # 茶碱
    "CFF","CAF",                                       # 咖啡因
    "TET","TET1",                                      # 四环素
    "PAR","PABA",                                      # 对氨基苯甲酸
    "BIO",                                             # 生物素
    "HEM","HEM1","HEC","HAS","HEA",                    # 血红素
    "NPM","FAM","FLC",                                 # 荧光团
    "DFH","DFH1","DFHB",                               # DFHBI (Spinach)
    "TBT","TO1","TO3",                                 # TO1-biotin (Mango) 等噻唑橙类
    "BTA","DAPI","PIC",                                # 其他染料
    "GAL","MAN","GLC","NGA","NAC",                     # 糖类（有些是功能配体）
    # 抗生素（RNA 靶标类）
    "NEO","TOB","KAN","GEN","STR","PARO","CHL","ERY",
    "SPM","SPD","PUT","AMD","LIN","CLI","TET",
    "PAR","NAG",                                       # 帕罗米星等
    "AMD",                                             # 放线菌素 D
    "HIS",                                             # 组氨酸适配体
    "DAP",                                             # 二氨基丙酸（赖氨酸核糖开关探针）
    "PCP","PCT",                                       # 帕罗西汀?
    "OCT","DOX","DAU",                                 # 道诺霉素等嵌入剂（RNA 结合）
    "EBW","ETB",                                       # 溴化乙锭
}

def parse_coords(path):
    """解析坐标：返回 (atoms, chain_res_map)
    atoms: dict[(chain, resname, resnum, icode)] = {atom_name: (x,y,z)}  # 用 list 更快
    简化：返回按链聚合的原子数组 + 残基索引
    """
    # 数据格式: 每链 {residue_key: {atom: coord}}
    chains = defaultdict(dict)   # chain -> {reskey -> {atom_name: (x,y,z)}}
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[16] not in (" ", "A"):  # 跳过 altloc 副构象
                continue
            chain = line[21]
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            icode = line[26].strip()
            atom = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            chains[chain].setdefault((resname, resnum, icode), {})[atom] = (x, y, z)
    return dict(chains)

def chain_center_atoms(chains, resnames_set):
    """提取某链中属于给定残基名集合的原子坐标（扁平数组）"""
    pts = []
    for reskey, atoms in chains.items():
        if reskey[0] in resnames_set:
            pts.extend(atoms.values())
    return np.array(pts, dtype=float).reshape(-1, 3) if pts else np.empty((0, 3))

def min_dist_to(a_pts, b_pts):
    """a_pts(Nx3) 到 b_pts(Mx3) 的最小距离（cKDTree，大结构也快）"""
    if len(a_pts) == 0 or len(b_pts) == 0:
        return None
    tree = cKDTree(b_pts)
    d, _ = tree.query(a_pts, k=1)
    return float(d.min())

def classify_chain_type(atom_resnames):
    names = set(atom_resnames)
    has_rna = any(n in RNA_RES for n in names)
    has_dna = any(n in DNA_RES for n in names)
    has_aa  = any(n in AA_RES for n in names)
    if has_aa:
        return "mixed" if (has_rna or has_dna) else "protein"
    if has_rna and has_dna:
        return "mixed_nuc"
    if has_rna:
        return "rna"
    if has_dna:
        return "dna"
    return "other"

def analyze_entry(pdb_id):
    path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    try:
        chains = parse_coords(path)
    except Exception as e:
        return {"pdb": pdb_id, "error": str(e)}

    # 链类型
    ctypes = {}
    for ch, res in chains.items():
        ctypes[ch] = classify_chain_type([rk[0] for rk in res])

    rna_chains = {ch for ch, t in ctypes.items() if t == "rna"}
    prot_chains = {ch for ch, t in ctypes.items() if t == "protein"}
    mixed_chains = {ch for ch, t in ctypes.items() if t == "mixed"}

    # RNA 原子（所有 RNA 链）与蛋白原子（蛋白+混合链）
    rna_atoms_all = []
    prot_atoms_all = []
    for ch in rna_chains:
        rna_atoms_all.append(chain_center_atoms(chains[ch], RNA_RES))
    for ch in prot_chains | mixed_chains:
        prot_atoms_all.append(chain_center_atoms(chains[ch], AA_RES))
    rna_pts = np.vstack(rna_atoms_all) if rna_atoms_all else np.empty((0, 3))
    prot_pts = np.vstack(prot_atoms_all) if prot_atoms_all else np.empty((0, 3))

    # 建树（每条目一次）
    rna_tree = cKDTree(rna_pts) if len(rna_pts) else None
    prot_tree = cKDTree(prot_pts) if len(prot_pts) else None

    # 蛋白-RNA 结合态
    prot_rna_dist = min_dist_to(prot_pts, rna_pts) if len(prot_pts) and len(rna_pts) else None
    rna_protein_bound = prot_rna_dist is not None and prot_rna_dist < CONTACT_CUTOFF

    # 功能配体（other 类）逐残基接触判定
    ligand_contacts = []   # {ligand, chain, min_to_rna, min_to_prot, binds_rna, binds_prot}
    seen = set()
    for ch, res in chains.items():
        for reskey, atoms in res.items():
            rn = reskey[0]
            if rn in RNA_RES or rn in AA_RES or rn in DNA_RES:
                continue  # 非配体
            key = (rn, ch, reskey[1])
            if key in seen:
                continue
            seen.add(key)
            pts = np.array(list(atoms.values()), dtype=float).reshape(-1, 3)
            if len(pts) == 0:
                continue
            if rna_tree is not None:
                d_rna, _ = rna_tree.query(pts, k=1)
                d_rna = float(d_rna.min())
            else:
                d_rna = None
            if prot_tree is not None:
                d_prot, _ = prot_tree.query(pts, k=1)
                d_prot = float(d_prot.min())
            else:
                d_prot = None
            ligand_contacts.append({
                "ligand": rn, "chain": ch, "resnum": reskey[1],
                "min_to_rna": d_rna, "min_to_prot": d_prot,
                "binds_rna": d_rna is not None and d_rna < CONTACT_CUTOFF,
                "binds_prot": d_prot is not None and d_prot < CONTACT_CUTOFF,
            })

    return {
        "pdb": pdb_id,
        "n_rna_chains": len(rna_chains),
        "n_prot_chains": len(prot_chains),
        "rna_protein_bound": rna_protein_bound,
        "prot_rna_min_dist": prot_rna_dist,
        "ligands": ligand_contacts,
    }

def main():
    candidates = json.load(open(os.path.join(OUT_DIR, "candidates.json")))
    ids = [c["pdb"] for c in candidates]
    print(f"第 2 阶段: 分析 {len(ids)} 个候选条目", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(ex.map(analyze_entry, ids, chunksize=8)):
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"进度: {i+1}/{len(ids)}", flush=True)

    # 汇总
    n_ok = sum(1 for r in results if "error" not in r)
    n_rna_ligand_bound = 0
    n_rna_prot_bound = 0
    n_rna_ligand_bound_known = 0
    lig_bind_freq = defaultdict(int)
    known_lig_entries = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        binds = [l for l in r["ligands"] if l["binds_rna"]]
        if binds:
            n_rna_ligand_bound += 1
            for l in binds:
                lig_bind_freq[l["ligand"]] += 1
                if l["ligand"] in KNOWN_TARGET_LIGANDS:
                    known_lig_entries[l["ligand"]].append(r["pdb"])
        if r["rna_protein_bound"]:
            n_rna_prot_bound += 1

    summary = {
        "total_candidates": len(ids),
        "analyzed": n_ok,
        "rna_ligand_bound_entries": n_rna_ligand_bound,
        "rna_protein_bound_entries": n_rna_prot_bound,
        "ligand_bind_freq": dict(sorted(lig_bind_freq.items(), key=lambda x: -x[1])),
        "known_target_ligand_entries": {k: v for k, v in known_lig_entries.items()},
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "contacts_summary.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(results, open(os.path.join(OUT_DIR, "contacts.json"), "w"), indent=1, ensure_ascii=False)

    print(f"\n===== 空间接触分析（第 2 阶段） =====", flush=True)
    print(f"候选条目: {len(ids)}  分析成功: {n_ok}", flush=True)
    print(f"配体接触 RNA (<5Å) 条目: {n_rna_ligand_bound}", flush=True)
    print(f"蛋白接触 RNA (<5Å) 条目: {n_rna_prot_bound}", flush=True)
    print(f"Top 20 结合 RNA 的配体: ", flush=True)
    for rn, c in list(summary["ligand_bind_freq"].items())[:20]:
        print(f"  {rn:6s} {c}", flush=True)
    print(f"已知靶标配体命中条目数: {sum(len(v) for v in known_lig_entries.values())}", flush=True)
    print(f"结果写入: {OUT_DIR}/contacts.json, contacts_summary.json", flush=True)

if __name__ == "__main__":
    main()
