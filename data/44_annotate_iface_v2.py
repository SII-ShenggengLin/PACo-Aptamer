#!/usr/bin/env python3
"""第 44 步：为 eval_inputs_v2 补 interface_chains 标注（14 号逻辑，v2 数据）

42 号脚本只存了 min_dist；界面恢复率指标需要逐残基接触标注。
对 A 层（indep ∪ zero_shot）条目：配体 5.0Å 接触 RNA 残基 → interface_chains
写回 eval_inputs_v2/{pdb}.json（新增字段），不重建文件。
"""
import os, json
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA, "rcsb_pdb")
OUT_DIR = os.path.join(DATA, "survey", "eval_inputs_v2")
CONTACT = 5.0
STD = {"A", "C", "G", "U"}

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
    "ZDU": "U", "ZGU": "U", "ZYD": "U", "ZYT": "U", "I": "U",
}

def extract_rna(pdb_id):
    """{chain: [(resname, resnum, icode, {atom: coord})]}，14 号同款"""
    path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return None
    chains = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip()
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
    result = {}
    for chain, atoms in chains.items():
        residues = defaultdict(dict)
        for resname, resnum, icode, atom, coord in atoms:
            residues[(resname, resnum, icode)][atom] = coord
        std_count = sum(1 for rn, _, _ in residues if rn in STD or rn in MOD_MAP)
        if std_count == 0:
            continue
        result[chain] = sorted(residues.items(),
                               key=lambda kv: (int(kv[0][1]) if kv[0][1].isdigit() else 0, kv[0][2]))
    return result

def load_ccd_other():
    cands = json.load(open(os.path.join(DATA, "survey", "candidates_ccd.json")))
    s = set()
    for c in cands:
        s.update(c["ligands"].get("other", []))
    return s

def main():
    sp = json.load(open(os.path.join(DATA, "splits_v2", "splits_v2.json")))
    A_entries = sorted({m.split("_")[0] for m in
                        set(sp["independent_test"]) | set(sp["zero_shot_targets"])})
    func_ligs = load_ccd_other()
    print(f"A 层条目: {len(A_entries)}", flush=True)

    n_iface = 0
    for pdb_id in A_entries:
        jp = os.path.join(OUT_DIR, f"{pdb_id}.json")
        e = json.load(open(jp))
        # 先清掉历史(可能来自索引错位旧版)的 interface_chains，本脚本全量重算
        for c in e.get("ligand_contacts", []):
            c.pop("interface_chains", None)
        chains = extract_rna(pdb_id)
        if not chains:
            continue
        rna_atoms = {}
        for ch, reslist in chains.items():
            pts = []
            for (_, _, _), atoms in reslist:
                pts.extend(atoms.values())
            rna_atoms[ch] = np.array(pts, dtype=float).reshape(-1, 3)

        lig_coords = defaultdict(list)
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

        all_rna_pts = np.vstack(list(rna_atoms.values())) if rna_atoms else np.empty((0, 3))
        tree_all = cKDTree(all_rna_pts) if len(all_rna_pts) else None
        new_contacts = []
        for (rn, ch, rnum), coords in lig_coords.items():
            pts = np.array(coords, dtype=float)
            if tree_all is None:
                continue
            d, _ = tree_all.query(pts, k=1)
            mind = float(d.min())
            if mind >= CONTACT:
                continue
            touched = defaultdict(set)
            for chain, arr in rna_atoms.items():
                if len(arr) == 0:
                    continue
                # query 返回 (dist, idx)：二者都按查询点(配体原子)排序；
                # RNA 原子下标在第二个返回值里。曾误用 np.where(dt<CONTACT)[0]
                # (配体原子下标) 当 RNA 原子下标 → 残基整体错位(2026-08-19 修复)
                dt, idx = cKDTree(arr).query(pts, k=1)
                for j in idx[dt < CONTACT]:
                    touched[chain].add(int(j))
            interface_chains = {}
            for chain, atom_ids in touched.items():
                res_ranges = []
                off = 0
                for (rn2, rn_num, ic), atoms in chains[chain]:
                    res_ranges.append((off, off + len(atoms), rn_num, ic))
                    off += len(atoms)
                hit = set()
                for ai in atom_ids:
                    for start, end, rn_num, ic in res_ranges:
                        if start <= ai < end:
                            hit.add((rn_num, ic))
                            break
                interface_chains[chain] = sorted(hit)
            # 合并到既有 contact（min_dist 已有）——只补 interface_chains；
            # 同 (ligand, chain) 多实例(多结合位点)取残基并集而非覆盖
            found = next((c for c in e.get("ligand_contacts", [])
                          if c.get("ligand") == rn and c.get("chain") == ch), None)
            if found is not None:
                prev = {k: {tuple(r) for r in v}
                        for k, v in found.get("interface_chains", {}).items()}
                new = {k: {tuple(r) for r in v} for k, v in interface_chains.items()}
                found["interface_chains"] = {
                    k: sorted(prev.get(k, set()) | new.get(k, set()))
                    for k in prev.keys() | new.keys()}
                n_iface += 1
            else:
                new_contacts.append({"ligand": rn, "chain": ch, "min_dist": round(mind, 2),
                                     "interface_chains": interface_chains})
        if new_contacts:
            e.setdefault("ligand_contacts", []).extend(new_contacts)
        json.dump(e, open(jp, "w"), indent=1, ensure_ascii=False)

    # manifest 同步（保持单一事实来源）
    manifest = json.load(open(os.path.join(DATA, "survey", "eval_manifest_v2.json")))
    for entry in manifest:
        if entry["pdb"] not in A_entries:
            continue
        jp = os.path.join(OUT_DIR, f"{entry['pdb']}.json")
        updated = json.load(open(jp))
        if updated["ligand_contacts"]:
            entry["ligand_contacts"] = updated["ligand_contacts"]
    json.dump(manifest, open(os.path.join(DATA, "survey", "eval_manifest_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"完成：{n_iface} 个配体接触补充 interface_chains 标注", flush=True)

if __name__ == "__main__":
    main()
