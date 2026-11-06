#!/usr/bin/env python3
"""第 51 步：splits_v2 权威链库（canonical）构建——统一序列与结构真值

动机（2026-08-16 审计发现）：
- 同一 test 链在不同方法的 B 层评估中 native 长度不一致（40/477 链 ±1~2，3 链严重：
  PACo 用 s2_samples 的多拷贝拼接序列 425/250/290nt 当真值；RhoDesign 用 43c 的
  每残基仅 P 原子的坏文件；gRNAde 用 43b 未去重的多拷贝文件混拷贝坐标）。
- seqs_all.json 对部分链有提取瑕疵（含 'X'、边缘偏移）。
- 29 条 s2 独有链实为蛋白骨架痕迹链（非 RNA，全在 train）。

规则：
- 每条链一个权威 (PDB, 序列) 对：model 1 + altloc 过滤 + 同链 (resnum,icode) 首拷贝
  （key 非相邻重复即截断——多拷贝晶体/NMR 只取第一拷贝）。
- 残基→碱基：标准 ACGU > 修饰核苷酸 MOD_MAP > CCD mon_nstd_parent_comp_id（递归）>
  DNA DA/DC/DG/DT→A/C/G/U；均失败 → 整条链标记 unmapped 剔除（不静默丢残基）。
- 输出：data/canonical_v2/{stem}.pdb（标准固定列）+ survey/canonical_v2.json
  （seq/n_res/src/qc 标记）+ survey/canonical_v2_dropped.json（剔除链及原因）。
- 与 seqs_all 不一致处记录 library_mismatch（canonical 为准）。
"""
import os, json, gzip

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT_DIR = os.path.join(DATA_DIR, "canonical_v2")
os.makedirs(OUT_DIR, exist_ok=True)

import importlib.util
spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}
DNA_MAP = {"DA": "A", "DC": "C", "DG": "G", "DT": "U", "DU": "U"}


def load_ccd_parent():
    """components.cif.gz → {comp_id: parent_comp_id}（仅收有 parent 的）"""
    parent = {}
    cur_id = None
    with gzip.open(os.path.join(DATA_DIR, "ccd", "components.cif.gz"), "rt") as f:
        for line in f:
            if line.startswith("_chem_comp.id"):
                cur_id = line.split(None, 1)[1].strip()
            elif line.startswith("_chem_comp.mon_nstd_parent_comp_id"):
                p = line.split(None, 1)[1].strip()
                if cur_id and p not in ("?", "."):
                    parent[cur_id] = p.strip('"')
    return parent


CCD_PARENT = load_ccd_parent()


def base_of(rn):
    """残基名 → 碱基（ACGU/N），无法映射返回 None
    特例：resname 'N' = CCD RNA LINKING 未指明核苷酸 → 字母 'N'（未知碱基，保留链）；
    '+A/+C/+G/+U/+I' = 5' 端磷酸化变体（旧 PDB 惯例）→ 对应碱基"""
    if rn in STD:
        return rn
    if rn == "N":
        return "N"
    if rn.startswith("+") and len(rn) == 2 and rn[1] in "ACGUI":
        return rn[1] if rn[1] != "I" else "G"
    if rn in MOD_MAP:
        return MOD_MAP[rn]
    if rn in DNA_MAP:
        return DNA_MAP[rn]
    seen = set()
    cur = rn
    while cur in CCD_PARENT and cur not in seen:
        seen.add(cur)
        cur = CCD_PARENT[cur]
        if cur in STD or cur == "N":
            return cur if cur != "N" else "N"
        if cur in MOD_MAP:
            return MOD_MAP[cur]
        if cur in DNA_MAP:
            return DNA_MAP[cur]
    return None


def fmt_atom(serial, atom, altloc, rn, chain, resnum, icode, x, y, z, occ, bfac, elem):
    # 原子名规范化：旧式 'O5*' → "O5'"（PDB 2.x 命名）；标准 PDB 右对齐（单字符元素名前导空格）
    atom = atom.replace("*", "'")
    if len(atom) < 4 and atom[0].isalpha():
        atom_f = (" " + atom).ljust(4)
    else:
        atom_f = atom.ljust(4)[:4]
    return (f"ATOM  {serial:5d} {atom_f}{altloc:1s}{rn:>3s} {chain:1s}{resnum:>4s}{icode:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}  \n")


def canonicalize(stem):
    """返回 (pdb_content, seq, n_res, qc_flags) 或 (None, reason, None, None)"""
    src = None
    for d in ("chain_pdb", "chain_pdb_v2"):
        p = os.path.join(DATA_DIR, d, f"{stem}.pdb")
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return None, "no_pdb", None, None
    groups = []
    cur = None
    seen_keys = set()
    with open(src) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            if cur is not None and key == cur[0]:
                cur[1].append(line)
                continue
            if key in seen_keys:
                # 同链 (resnum,icode) 非相邻重复 = 第二拷贝起点（多拷贝晶体）→ 截断
                cur = "STOP"
                break
            seen_keys.add(key)
            cur = (key, [line])
            groups.append(cur)
    if not groups:
        return None, "no_atoms", None, None
    # 残基映射（任一不可映射 → 整链失败，不静默丢残基）
    bases = []
    for _, ls in groups:
        b = base_of(ls[0][17:20].strip())
        if b is None:
            return None, f"unmapped_residue:{ls[0][17:20].strip()}", None, None
        bases.append(b)
    seq = "".join(bases)
    # 重写标准格式
    out_lines = []
    serial = 1
    for (_, ls) in groups:
        for line in ls:
            atom = line[12:16].strip()
            rn = line[17:20].strip()
            resnum = line[22:26].strip()
            icode = line[26].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            occ = line[54:60].strip()
            bfac = line[60:66].strip()
            elem = line[76:78].strip() if len(line) >= 78 else atom.strip()[0]
            out_lines.append(fmt_atom(serial, atom, " ", rn, "A", resnum, icode,
                                      x, y, z, float(occ) if occ else 1.0,
                                      float(bfac) if bfac else 20.0, elem))
            serial += 1
    out_lines.append("END\n")
    # QC
    qc = []
    atoms_per_res = {}
    for line in out_lines:
        if not line.startswith("ATOM"):
            continue
        k2 = (line[22:26].strip(), line[26].strip())
        atoms_per_res.setdefault(k2, set()).add(line[12:16].strip())
    n_bb = sum(1 for a in atoms_per_res.values() if "C1'" in a and "C4'" in a)
    if n_bb < len(groups) * 0.5:
        qc.append("backbone_incomplete")
    if len(seq) < 5:
        qc.append("too_short")
    return "".join(out_lines), seq, len(groups), qc


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    seqs_all = json.load(open(os.path.join(DATA_DIR, "splits_v2", "seqs_all.json")))
    chains = sorted(set(sp["train"]) | set(sp["val"]) | set(sp["test"]))
    print(f"目标链（train+val+test）: {len(chains)}", flush=True)
    meta, dropped = {}, {}
    n_mismatch = 0
    for i, stem in enumerate(chains):
        content, seq_or_reason, n_res, qc = canonicalize(stem)
        if content is None:
            dropped[stem] = seq_or_reason
            continue
        with open(os.path.join(OUT_DIR, f"{stem}.pdb"), "w") as f:
            f.write(content)
        entry = {"seq": seq_or_reason, "n_res": n_res, "qc": qc}
        sa = seqs_all.get(stem)
        if sa is not None and sa.replace("T", "U") != seq_or_reason:
            entry["library_mismatch"] = {"seqs_all_len": len(sa), "canonical_len": len(seq_or_reason)}
            n_mismatch += 1
        meta[stem] = entry
        if (i + 1) % 1000 == 0:
            print(f"进度 {i+1}/{len(chains)}", flush=True)
    json.dump(meta, open(os.path.join(DATA_DIR, "survey", "canonical_v2.json"), "w"))
    json.dump(dropped, open(os.path.join(DATA_DIR, "survey", "canonical_v2_dropped.json"), "w"), indent=1)
    print(f"\n完成: {len(meta)}/{len(chains)} 成功, 剔除 {len(dropped)}, library 不一致 {n_mismatch}", flush=True)
    # 划分覆盖核对
    for k in ("train", "val", "test"):
        got = sum(1 for c in sp[k] if c in meta)
        print(f"  {k}: {got}/{len(sp[k])}", flush=True)
    if dropped:
        import collections
        reasons = collections.Counter(v.split(":")[0] for v in dropped.values())
        print("  剔除原因:", dict(reasons), flush=True)


if __name__ == "__main__":
    main()
