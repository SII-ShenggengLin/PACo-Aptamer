#!/usr/bin/env python3
"""42：v2 评估输入构建（新划分 v2-B）
- 评估集：test（B 层 477）+ indep ∪ zero_shot（A 层 289）→ 712 链 / 528 条目
- 序列：chain_pdb（BGSU 单链文件）+ s2 独有 11 链（从 s2_samples 导出 PDB）
- 配体/界面：rcsb_pdb 条目 → CCD other 类配体 → 5.0Å 接触界面（同 14 号口径）
产出：survey/eval_inputs_v2/（每条目 JSON）+ eval_manifest_v2.json
"""
import os, re, json
import importlib.util
import numpy as np
from scipy.spatial import cKDTree
import biotite.structure.io.pdb as pdbio

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA, "rcsb_pdb")
OUT_DIR = os.path.join(DATA, "survey", "eval_inputs_v2")
CP2_DIR = os.path.join(DATA, "chain_pdb_v2")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CP2_DIR, exist_ok=True)
CONTACT = 5.0

# 修饰核苷酸 → 母碱基映射（32v2/44/69 号同一张表，单一事实来源）
_spec = importlib.util.spec_from_file_location(
    "m32v2", os.path.join(DATA, "32v2_compute_metrics_nampnn.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
MOD_MAP = _m.MOD_MAP

sp = json.load(open(os.path.join(DATA, "splits_v2", "splits_v2.json")))
cands = {c["pdb"]: c for c in json.load(open(os.path.join(DATA, "survey", "candidates.json")))}
s2 = torch_load = __import__("torch").load(os.path.join(DATA, "s2_btrain", "s2_samples.pt"), map_location="cpu")
G27 = ["P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
       "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
       "N6", "N7", "C8", "N9", "OP1", "OP2"]
ELEM = {"C": "C", "O": "O", "N": "N", "P": "P", "S": "S"}

# 1. s2 独有链导出 PDB（无 chain_pdb 文件的 11 条）
s2_only = [m for m in set(sp["test"]) | set(sp["independent_test"]) | set(sp["zero_shot_targets"])
           if m in s2 and not os.path.exists(os.path.join(DATA, "chain_pdb", m + ".pdb"))]
print(f"s2 独有链导出: {len(s2_only)}", flush=True)
for stem in s2_only:
    d = s2[stem]
    coords = np.array(d["coords_27"])   # (L, 27, 3)
    seq = d["sequence"]
    L = coords.shape[0]
    lines = [f"ATOM  {i+1:5d}  {G27[a]:4s} {seq[i]:3s} A{i+1:4d}    "
             f"{coords[i,a,0]:8.3f}{coords[i,a,1]:8.3f}{coords[i,a,2]:8.3f}  1.00 20.00           "
             f"{ELEM.get(G27[a][0], 'C'):2s}\n"
             for i in range(L) for a in range(27)]
    with open(os.path.join(CP2_DIR, stem + ".pdb"), "w") as f:
        f.writelines(lines)
print("导出完成", flush=True)

# 2. 评估条目 → 链映射
eval_chains = sorted(set(sp["test"]) | set(sp["independent_test"]) | set(sp["zero_shot_targets"]))
by_entry = {}
for m in eval_chains:
    e = m.split("_")[0]
    by_entry.setdefault(e, []).append(m)

def parse_entry(pdb_path):
    """解析 rcsb_pdb 条目：返回 {chain: {resname_seq, coords_chain}, ligands: {lig: coords}}"""
    arr = pdbio.PDBFile.read(pdb_path).get_structure()[0]
    out = {"chains": {}, "ligands": {}}
    for ch in set(arr.chain_id):
        a = arr[arr.chain_id == ch]
        het = a.hetero
        rna = ~het
        # RNA 残基序列（P 原子代表）
        ap = a[rna & (a.atom_name == "P")]
        if ap.array_length() > 0:
            seq = []
            for rn in ap.res_name:
                rn = rn.strip().upper()
                b = "X"
                for x in "ACGU":
                    if rn == x or rn.endswith(x):
                        b = x; break
                seq.append(b)
            out["chains"][ch] = "".join(seq)
        # 配体（hetero 且非水/修饰核苷酸）
        for lch in set(a[het].chain_id):
            la = a[het & (a.chain_id == lch)]
            if la.array_length() == 0: continue
            rns = set(la.res_name)
            for rn in rns:
                rn = str(rn).strip()
                if rn == "HOH" or len(rn) <= 1: continue
                if rn in MOD_MAP: continue          # 修饰核苷酸不是配体（曾为死代码，2026-08-19 修复）
                if rn in {"A","C","G","U"}:
                    continue   # 游离核苷酸不算配体
                larr = la[la.res_name == rn]
                out["ligands"].setdefault(rn, {"coords": larr.coord, "chain": lch})
    return out

# 3. 逐条目构建
manifest = []
n_ok = 0
for e in sorted(by_entry):
    src = os.path.join(PDB_DIR, e + ".pdb")
    if not os.path.exists(src):
        print(f"  缺 rcsb_pdb: {e}", flush=True)
        continue
    p = parse_entry(src)
    seqs = {m.split("_")[-1]: p["chains"].get(m.split("_")[-1]) for m in by_entry[e]}
    # 部分链在 chain_pdb 更可靠 → 兜底读 chain_pdb_v2/chain_pdb
    for m in by_entry[e]:
        ch = m.split("_")[-1]
        if not seqs.get(ch):
            for f in (os.path.join(DATA, "chain_pdb", m + ".pdb"),
                      os.path.join(CP2_DIR, m + ".pdb")):
                if os.path.exists(f):
                    try:
                        arr = pdbio.PDBFile.read(f).get_structure()[0]
                        ap = arr[(~arr.hetero) & (arr.atom_name == "P")]
                        seq = ""
                        for r in ap.res_name:
                            rn = r.strip().upper()
                            b = "X"
                            for x in "ACGU":          # 与主路径同款映射（修饰残基 → 母碱基）
                                if rn == x or rn.endswith(x):
                                    b = x; break
                            seq += b
                        seqs[ch] = seq
                    except Exception:
                        pass
                    break
    # 配体接触（cKDTree 5.0Å）
    lig_contacts = []
    for lig, info in p["ligands"].items():
        lc = info["coords"]
        for ch, seq in seqs.items():
            if not seq: continue
            # 该链原子坐标（从条目解析的 RNA 原子）
            arr = pdbio.PDBFile.read(src).get_structure()[0]
            a = arr[(arr.chain_id == ch) & (~arr.hetero)]
            if a.array_length() == 0: continue
            tree = cKDTree(a.coord)
            d, _ = tree.query(lc)
            if d.min() < CONTACT:
                lig_contacts.append({"ligand": lig, "chain": ch, "min_dist": float(d.min())})
    entry_json = {
        "pdb": e,
        "release_date": cands.get(e, {}).get("release_date", ""),
        "bound_ligands": sorted({c["ligand"] for c in lig_contacts}),
        "sequences": {ch: s for ch, s in seqs.items() if s},
        "ligand_contacts": lig_contacts,
        "splits_v2": [g for g in ("test", "independent_test", "zero_shot_targets")
                      if any(m in sp[g] for m in by_entry[e])],
    }
    with open(os.path.join(OUT_DIR, e + ".json"), "w") as f:
        json.dump(entry_json, f, indent=1)
    manifest.append(entry_json)
    n_ok += 1

json.dump(manifest, open(os.path.join(DATA, "survey", "eval_manifest_v2.json"), "w"), indent=1)
n_chains = sum(len(m["sequences"]) for m in manifest)
print(f"条目 {n_ok}/{len(by_entry)} | 链 {n_chains}", flush=True)
print("完成", flush=True)
