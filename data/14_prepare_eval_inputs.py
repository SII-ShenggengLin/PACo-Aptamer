#!/usr/bin/env python3
"""第 14 步：准备评估输入数据（independent_test + zero_shot_targets）

对每个评估条目：
1. 从 rcsb_pdb 提取 RNA 链序列（修饰核苷酸归一化到母碱基，A/C/G/U）
2. 提取功能配体（CCD other 类）坐标
3. 计算界面标注：配体重原子 5.0Å 内 RNA 残基 = 口袋残基
4. 输出 eval_manifest.json（总表）+ eval_inputs/（每条目 JSON）

评估口径（防泄漏第 6 条）：只输出序列与几何，不含任何预测器输入痕迹。
"""
import os, re, json
from collections import defaultdict
from scipy.spatial import cKDTree
import numpy as np

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
OUT_DIR = os.path.join(DATA_DIR, "survey", "eval_inputs")
os.makedirs(OUT_DIR, exist_ok=True)
CONTACT = 5.0

# 修饰核苷酸 → 母碱基（用于序列归一化；来自 CCD RNA linking 子集的手工映射）
MOD_MAP = {
    "1MA": "A", "MIA": "A", "6MA": "A", "2MA": "A", "M2A": "A", "M6A": "A",
    "M7A": "A", "MMU": "A", "ZAD": "A", "A23": "A", "A2M": "A", "N6G": "A",
    "5MC": "C", "M5C": "C", "CMO": "C", "CCC": "C", "CBR": "C", "C2L": "C",
    "C43": "C", "C5L": "C", "C7M": "C", "CD0": "C", "CFL": "C", "CG1": "C",
    "CH6": "C", "CNO": "C", "CR2": "C", "CRB": "C", "CSL": "C", "TCY": "C",
    "2MG": "G", "7MG": "G", "M2G": "G", "G7M": "G", "MAG": "G", "GDO": "G",
    "G25": "G", "G2L": "G", "G36": "G", "G37": "G", "G49": "G", "GH3": "G",
    "GHT": "G", "GIA": "G", "GMU": "G", "GNG": "G", "GOM": "G", "GSO": "G",
    "GTP": "G", "NMG": "G", "OGU": "G", "QUO": "G", "YYG": "G", "YG": "G",
    "YOF": "G", "O2G": "G", "O3G": "G", "O8G": "G", "PRF": "G", "7GU": "G",
    "PSU": "U", "5MU": "U", "H2U": "U", "OMU": "U", "UR3": "U", "20U": "U",
    "4SU": "U", "M5U": "U", "5OHU": "U", "5BU": "U", "DHU": "U", "S4U": "U",
    "M3U": "U", "MNU": "U", "U2L": "U", "U25": "U", "U31": "U", "U36": "U",
    "UAR": "U", "UBD": "U", "UCL": "U", "UFR": "U", "UFT": "U", "UMA": "U",
    "UMC": "U", "UML": "U", "UMO": "U", "UOX": "U", "UPA": "U", "UPL": "U",
    "UPT": "U", "URD": "U", "URM": "U", "URO": "U", "US1": "U", "US5": "U",
    "USE": "U", "USM": "U", "UTR": "U", "XUA": "U", "ZTH": "U", "I4U": "U",
    "DHE": "U", "H3U": "U", "HBI": "U", "HDU": "U", "HIA": "U", "HMU": "U",
    "HPU": "U", "HTA": "U", "IMC": "U", "KAG": "U", "LHG": "U", "M3U": "U",
    "MA6": "U", "MAM": "U", "MAP": "U", "MAQ": "U", "MAU": "U", "MBG": "U",
    "MDO": "U", "MED": "U", "MG3": "U", "MGU": "U", "MIS": "U", "MO6": "U",
    "MOG": "U", "MOW": "U", "MPG": "U", "MPN": "U", "MTU": "U", "MVA": "U",
    "N2G": "U", "NC6": "U", "NRQ": "U", "O4U": "U", "O5U": "U", "OA1": "U",
    "ONH": "U", "ONL": "U", "ONP": "U", "ORD": "U", "OTG": "U", "P5U": "U",
    "PBG": "U", "PGP": "U", "PGV": "U", "PHU": "U", "PMP": "U", "PPN": "U",
    "PR5": "U", "PSD": "U", "PTR": "U", "PYO": "U", "R5P": "U", "S2C": "U",
    "S2G": "U", "S6G": "U", "SEL": "U", "SGP": "U", "SHP": "U", "SMP": "U",
    "SRA": "U", "SRS": "U", "SST": "U", "STU": "U", "SUL": "U", "T6A": "U",
    "TAF": "U", "TBR": "U", "TDR": "U", "TFA": "U", "TGP": "U", "TGT": "U",
    "THX": "U", "TLC": "U", "TMD": "U", "TMS": "U", "TRC": "U", "TRQ": "U",
    "TSP": "U", "TTS": "U", "TUR": "U", "TYP": "U", "U8O": "U", "UDP": "U",
    "UNK": "U", "VAS": "U", "WBU": "U", "XCP": "U", "YCM": "U", "ZCT": "U",
    "ZDU": "U", "ZGU": "U", "ZYD": "U", "ZYT": "U", "I": "U",  # 肌苷按 U 处理
}
# 标准
STD = {"A", "C", "G", "U"}
# 功能配体 = CCD other 类（12 号脚本产物）
CCD_OTHER = None

def load_ccd_other():
    """从 candidates_ccd.json 提取 CCD 判定为 other 的全部配体名"""
    global CCD_OTHER
    if CCD_OTHER is None:
        cands = json.load(open(os.path.join(DATA_DIR, "survey", "candidates_ccd.json")))
        s = set()
        for c in cands:
            s.update(c["ligands"].get("other", []))
        CCD_OTHER = s
    return CCD_OTHER

def extract_rna(pdb_id):
    """提取条目内全部 RNA 链：{chain: [(resname, resnum, icode, atoms)]}
    atoms = {atom: (x,y,z)}；只保留 5'→3' 主链顺序（按 resnum 排序）
    """
    path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return None
    chains = defaultdict(list)   # chain -> list of (resname, resnum, icode, {atom: coord})
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            # HETATM 只有修饰核苷酸才收入链（配体另算）
            if rec == "HETATM" and resname not in MOD_MAP:
                continue
            chain = line[21]
            resnum = line[22:26].strip()
            icode = line[26].strip()
            atom = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            chains[chain].append((resname, resnum, icode, atom, (x, y, z)))
    # 聚合残基
    result = {}
    for chain, atoms in chains.items():
        residues = defaultdict(dict)
        for resname, resnum, icode, atom, coord in atoms:
            residues[(resname, resnum, icode)][atom] = coord
        # 判断是否为 RNA 链（≥1 标准残基）
        std_count = sum(1 for rn, _, _ in residues if rn in STD or rn in MOD_MAP)
        if std_count == 0:
            continue
        result[chain] = sorted(residues.items(), key=lambda kv: (int(kv[0][1]) if kv[0][1].isdigit() else 0, kv[0][2]))
    return result

def to_seq(residues):
    """RNA 链残基 → 序列（修饰归一化），返回 (seq, mod_positions)"""
    seq = []
    mods = []
    for i, ((rn, rn_num, ic), _) in enumerate(residues):
        if rn in STD:
            seq.append(rn)
        elif rn in MOD_MAP:
            seq.append(MOD_MAP[rn])
            mods.append((i, rn))
        else:
            return None, None   # 含非 RNA 残基，跳过该链
    return "".join(seq), mods

def main():
    splits = json.load(open(os.path.join(DATA_DIR, "survey", "complex_splits.json")))
    func_ligs = load_ccd_other()
    eval_ids = sorted(set(splits["independent_test"]) | set(splits["zero_shot_targets"]))

    manifest = []
    skipped = []
    for pdb_id in eval_ids:
        chains = extract_rna(pdb_id)
        if not chains:
            skipped.append((pdb_id, "no_rna_chain"))
            continue

        # 收集：RNA 序列 + 全 RNA 原子（界面计算用）
        seqs = {}          # chain -> seq
        rna_atoms = {}     # chain -> np (N,3)
        valid = True
        for ch, reslist in chains.items():
            seq, mods = to_seq(reslist)
            if seq is None:
                valid = False
                break
            pts = []
            for (_, _, _), atoms in reslist:
                pts.extend(atoms.values())
            rna_atoms[ch] = np.array(pts, dtype=float).reshape(-1, 3)
            seqs[ch] = seq
        if not valid:
            skipped.append((pdb_id, "bad_rna_residues"))
            continue

        # 功能配体坐标 + 界面标注
        lig_coords = defaultdict(list)   # ligand -> [(chain, resnum, [coords])]
        with open(os.path.join(PDB_DIR, f"{pdb_id}.pdb")) as f:
            for line in f:
                if line[:6].strip() != "HETATM":
                    continue
                if line[16] not in (" ", "A"):
                    continue
                rn = line[17:20].strip()
                if rn not in func_ligs:
                    continue
                chain = line[21]
                resnum = line[22:26].strip()
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except ValueError:
                    continue
                lig_coords[(rn, chain, resnum)].append((x, y, z))

        # 界面：配体原子 → 最近 RNA 链原子 <5Å
        interface = {}   # chain -> [residx]
        ligand_contacts = []
        all_rna_pts = np.vstack(list(rna_atoms.values())) if rna_atoms else np.empty((0, 3))
        tree_all = cKDTree(all_rna_pts) if len(all_rna_pts) else None
        for (rn, ch, rnum), coords in lig_coords.items():
            pts = np.array(coords, dtype=float)
            d, _ = tree_all.query(pts, k=1)
            mind = d.min()
            if mind < CONTACT:
                # 找接触的 RNA 残基（按链；原子索引 → 残基索引正确映射）
                touched = defaultdict(set)
                for chain, arr in rna_atoms.items():
                    if len(arr) == 0:
                        continue
                    dt, _ = cKDTree(arr).query(pts, k=1)
                    for j in np.where(dt < CONTACT)[0]:
                        touched[chain].add(int(j))
                interface_chains = {}
                for chain, atom_ids in touched.items():
                    # 链内残基原子偏移表
                    res_ranges = []   # (start, end, residx)
                    off = 0
                    for ri, ((rn, rn_num, ic), atoms) in enumerate(chains[chain]):
                        res_ranges.append((off, off + len(atoms), rn_num, ic))
                        off += len(atoms)
                    hit = set()
                    for ai in atom_ids:
                        for start, end, rn_num, ic in res_ranges:
                            if start <= ai < end:
                                hit.add((rn_num, ic))
                                break
                    interface_chains[chain] = sorted(hit)
                ligand_contacts.append({"ligand": rn, "chain": ch,
                                        "min_dist": round(float(mind), 2),
                                        "interface_chains": interface_chains})

        entry = {
            "pdb": pdb_id,
            "release_date": splits["entry_info"][pdb_id]["release_date"],
            "bound_ligands": splits["entry_info"][pdb_id]["bound_ligands"],
            "sequences": seqs,
            "ligand_contacts": ligand_contacts,
        }
        json.dump(entry, open(os.path.join(OUT_DIR, f"{pdb_id}.json"), "w"), indent=1, ensure_ascii=False)
        manifest.append({"pdb": pdb_id, "n_chains": len(seqs),
                         "lengths": {k: len(v) for k, v in seqs.items()},
                         "ligands": splits["entry_info"][pdb_id]["bound_ligands"]})

    json.dump(manifest, open(os.path.join(OUT_DIR, "eval_manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(f"评估条目: {len(manifest)}（独立测试 {len(splits['independent_test'])} + zero_shot {len(splits['zero_shot_targets'])}）")
    print(f"跳过: {len(skipped)} {skipped[:5]}")
    print(f"输出: {OUT_DIR}/eval_manifest.json + 每条目 JSON")

if __name__ == "__main__":
    main()
