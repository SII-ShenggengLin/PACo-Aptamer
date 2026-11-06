#!/usr/bin/env python3
"""解析 PDB 文件：提取 RNA 链（分辨率<4Å，长度16-512nt）— AlignIF 复现第 2 步
逻辑：纯文本解析（快），识别纯 RNA 链（无蛋白残基、无 DNA），连续核苷酸段按链聚合。
"""
import os, re, json, sys
from collections import defaultdict

PDB_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data/rcsb_pdb"

# RNA 核苷酸残基（标准 + 常见修饰）；PDB 中 RNA 修饰残基为 M 开头或常见列表
RNA_RES = {"A", "C", "G", "U", "I",
           "1MA", "5MC", "5MU", "PSU", "2MG", "7MG", "M2G", "OMG", "OMC", "OMU",
           "MIA", "YYG", "H2U", "5BU", "20U", "UR3", "T6A", "BGH", "G7M", "QUO",
           "1MG", "2OM", "5NC", "6MA", "AMP", "CMP", "GMP", "UMP", "IPP", "GDP"}
# 氨基酸残基（标准 20）
AA_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
          "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
          "SEC","PYL","MSE","HOH"}
# 标准 DNA 残基
DNA_RES = {"DA", "DC", "DG", "DT", "DI"}

def parse_pdb(path):
    """返回 (resolution, chains) chains = {chain_id: [(residue, seqnum), ...]}"""
    resolution = None
    chains = defaultdict(list)
    cur_cur = None
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
                seqnum = line[22:26].strip()
                if line[16] not in (" ", "A"):  # 跳过 altloc 副构象
                    continue
                if (chain, seqnum) != cur_cur:
                    chains[chain].append((resname, seqnum))
                    cur_cur = (chain, seqnum)
            elif rec == "TER" or rec == "END":
                cur_cur = None
    return resolution, dict(chains)

def chain_type(residues):
    """判定链类型：rna / dna / protein / mixed"""
    names = {r[0] for r in residues}
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

def resolution_from_cif(cif_path):
    """从 mmCIF 读取分辨率（_refine.ls_d_res_high），cif2pdb 转换文件无 REMARK 2 时回退用"""
    try:
        with open(cif_path) as f:
            in_refine = False
            for line in f:
                if line.startswith("_refine.ls_d_res_high"):
                    val = line.split()[-1].strip()
                    try:
                        return float(val)
                    except ValueError:
                        return None
                if line.startswith("#") or (in_refine and not line.startswith("_")):
                    in_refine = False
    except Exception:
        return None
    return None

def main():
    out = []
    files = sorted(os.listdir(PDB_DIR))
    stats = defaultdict(int)
    n_entries = 0
    n_rna_entries = 0
    for i, fname in enumerate(files):
        if not fname.endswith(".pdb"):
            continue
        if i % 5000 == 0 and i > 0:
            print(f"进度: {i}/{len(files)}", flush=True)
        pdb_id = fname[:-4].upper()
        try:
            res, chains = parse_pdb(os.path.join(PDB_DIR, fname))
        except Exception as e:
            stats["parse_error"] += 1
            continue
        # cif2pdb 转换文件无 REMARK 2 → 从 cif 回退读分辨率
        if res is None:
            cif_path = os.path.join(os.path.dirname(PDB_DIR), "rcsb_cif", f"{pdb_id}.cif")
            if os.path.exists(cif_path):
                res = resolution_from_cif(cif_path)
        n_entries += 1
        entry_rna = False
        for chain, residues in chains.items():
            ctype = chain_type(residues)
            if ctype == "rna":
                n = len(residues)
                stats[f"rna_len_{'<=15' if n<=15 else ('16-512' if n<=512 else '>512')}"] += 1
                if res is not None and res <= 4.0 and 16 <= n <= 512:
                    stats["filtered_ok"] += 1
                    out.append({"pdb": pdb_id, "chain": chain, "length": n,
                                "resolution": res})
                    entry_rna = True
        if entry_rna:
            n_rna_entries += 1

    print(f"解析条目: {n_entries}  含RNA链条目: {n_rna_entries}")
    print(f"RNA链长度分布: {dict(stats)}")
    print(f"通过过滤(分辨率<4Å, 16-512nt): {stats['filtered_ok']} 条链")
    json.dump(out, open("/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data/chains_pass.json", "w"), indent=1)
    print(f"清单已写入 chains_pass.json")

if __name__ == "__main__":
    main()
