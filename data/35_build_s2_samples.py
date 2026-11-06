#!/usr/bin/env python3
"""第 35 步：S2 复合物训练样本构建（靶标条件化核心数据）

输入：complex_splits.json train 978 条目 + blayer_splits.json（泄漏剔除）
      rcsb_pdb/{pdb}.pdb（RNA 链 + 功能配体 + 界面 5.0Å）+ candidates_ccd.json
      chain_pdb/（训练池口径链，优先；缺失则从 rcsb_pdb 提取）
输出：data/s2_btrain/s2_samples.pt
      {链: {'sequence', 'coords_27' (L,27,3), 'ss', 'ligand_elements', 'ligand_coords',
            'interface_mask' (L,), 'pdb_id', 'chain', 'ligands'}}

设计要点（对齐基准结论）：
- 样本 = 链级 RNA + 条目级配体上下文（三图框架的 RNA 图 + 靶标图数据）
- 泄漏剔除：链 ∈ blayer val/test → 剔除（本审计发现 36+41=77 条重叠链）
- 界面：配体重原子 5.0Å 内 RNA 残基（与 14/28 号口径一致）
- 27 原子 = gRNAde RNA_ATOMS 顺序（与 S1 processed.pt 一致）
"""
import os, json, sys, torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from scipy.spatial import cKDTree

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")
from src.constants import RNA_ATOMS

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
CHAIN_DIR = os.path.join(DATA_DIR, "chain_pdb")
OUT_DIR = os.path.join(DATA_DIR, "s2_btrain")
os.makedirs(OUT_DIR, exist_ok=True)
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
FILL = 1e-5
CONTACT = 5.0

MOD_MAP = {  # 与 14/28 号脚本相同（修饰核苷酸 → 母碱基）
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
    "ZDU": "U", "ZGU": "U", "ZYD": "U", "ZYT": "U", "I": "U",
}
STD = {"A", "C", "G", "U"}
ATOM_IDX = {a: i for i, a in enumerate(RNA_ATOMS)}


def extract_rna(pdb_id, chain):
    """提取单链 RNA：[(resname, resnum, icode, {atom: coord})]，主链顺序"""
    residues = []
    with open(os.path.join(PDB_DIR, f"{pdb_id}.pdb")) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[21] != chain or line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            rnum = line[22:26].strip()
            ic = line[26].strip()
            atom = line[12:16].strip()
            if rec == "HETATM" and rn not in MOD_MAP:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            if not residues or residues[-1][1:3] != (rnum, ic):
                residues.append((rn, rnum, ic, {}))
            residues[-1][3][atom] = (x, y, z)
    seq_chars = []
    for rn, rnum, ic, atoms in residues:
        if rn in STD or rn in MOD_MAP:
            seq_chars.append(MOD_MAP.get(rn, rn))
    if not seq_chars:
        return None, None, None
    return residues, "".join(seq_chars), list(range(len(seq_chars)))


def coords_27(residues):
    """(L, 27, 3) gRNAde 顺序坐标，缺失 FILL"""
    L = len(residues)
    arr = np.full((L, len(RNA_ATOMS), 3), FILL, dtype=np.float32)
    for i, (rn, rnum, ic, atoms) in enumerate(residues):
        for atom, xyz in atoms.items():
            if atom in ATOM_IDX:
                arr[i, ATOM_IDX[atom]] = xyz
    return arr


def rnafold_ss(seq):
    import subprocess
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    import re
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None


def load_ccd_other():
    cands = json.load(open(os.path.join(DATA_DIR, "survey", "candidates_ccd.json")))
    s = set()
    for c in cands:
        s.update(c["ligands"].get("other", []))
    return s


def work_entry(args):
    pdb_id, func_ligs, exclude = args
    path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return {}
    # 链集合
    chains = set()
    for line in open(path):
        if line[:6].strip() == "ATOM":
            chains.add(line[21])
    # 配体坐标（功能配体）
    lig_elems, lig_coords = [], []
    lig_by_res = defaultdict(list)   # (rn, chain, resnum) -> [coords]
    for line in open(path):
        if line[:6].strip() != "HETATM" or line[16] not in (" ", "A"):
            continue
        rn = line[17:20].strip()
        if rn not in func_ligs:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        elem = line[76:78].strip() or "C"
        lig_elems.append(elem)
        lig_coords.append((x, y, z))
        lig_by_res[(rn, line[21], line[22:26].strip())].append((x, y, z))
    if not lig_elems:
        return {}
    lig_arr = np.array(lig_coords, dtype=np.float32)
    tree_lig = cKDTree(lig_arr)

    out = {}
    for ch in chains:
        stem = f"{pdb_id}_{ch}"
        if stem in exclude:
            continue
        residues, seq, _ = extract_rna(pdb_id, ch)
        if seq is None or not (16 <= len(seq) <= 512):
            continue
        coords = coords_27(residues)
        # 界面：残基原子 → 最近配体原子 <5Å
        interface = np.zeros(len(seq), dtype=bool)
        atom_off = 0
        for i, (rn, rnum, ic, atoms) in enumerate(residues):
            pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
            if len(pts):
                d, _ = tree_lig.query(pts, k=1)
                if d.min() < CONTACT:
                    interface[i] = True
        ss = rnafold_ss(seq)
        if ss is None or len(ss) != len(seq):
            continue
        out[stem] = {
            "sequence": seq,
            "coords_27": coords,
            "ss": ss,
            "ligand_elements": lig_elems,
            "ligand_coords": lig_arr,
            "interface_mask": interface,
            "pdb_id": pdb_id,
            "chain": ch,
        }
    return out


def main():
    splits = json.load(open(os.path.join(DATA_DIR, "survey", "complex_splits.json")))
    bl = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
    exclude = {k for k, v in bl.items() if v in ("val", "test")}   # 泄漏剔除
    func_ligs = load_ccd_other()
    print(f"train 条目 {len(splits['train'])}，剔除链 {len(exclude)}", flush=True)

    args_list = [(p, func_ligs, exclude) for p in splits["train"]]
    samples = {}
    skipped = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, res in enumerate(ex.map(work_entry, args_list)):
            samples.update(res)
            if (i + 1) % 200 == 0:
                print(f"进度: {i+1}/{len(args_list)} 样本 {len(samples)}", flush=True)
    torch.save(samples, os.path.join(OUT_DIR, "s2_samples.pt"))
    n_iface = sum(1 for s in samples.values() if s["interface_mask"].any())
    print(f"S2 样本: {len(samples)} 链（含界面 {n_iface}），写入 s2_samples.pt", flush=True)
    # 统计
    from collections import Counter
    ligs = Counter()
    for s in samples.values():
        for el in set(s["ligand_elements"]):
            ligs[el] += 1
    print(f"配体元素分布: {dict(ligs.most_common(5))}")


if __name__ == "__main__":
    main()
