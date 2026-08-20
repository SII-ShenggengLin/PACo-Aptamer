#!/usr/bin/env python3
"""第 64 步：B 层 v2 test 界面指标统一重建（canonical 真值）

与 57 号（A 层）同逻辑，差异：
- native = canonical_v2.json（'N' 位跳过）
- 输出池：grnade_out_btest_v2 / rhodesign_out_btest_v2（3 温度全池）/
  nampnn_out_btest_v2（t=0.1×10，缺 C1' 链用有效位）/ alignif_out_btest_v2（t=0.1×20，'-' 跳过）
- TACo 无序列池（38 号不保存序列）→ 不出界面指标（报告如实注明）
- 界面标注：eval_inputs_v2/{pdb}.json（44 号，配体 5.0Å），残基序号映射用 canonical PDB
输出：survey/interface_metrics_btest_v2.json（覆盖旧版——旧版为 pre-canonical 口径，作废）
"""
import os, sys, json, difflib
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
EI = os.path.join(SUR, "eval_inputs_v2")
CANON = os.path.join(DATA_DIR, "canonical_v2")

import importlib.util
spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}


def interface_positions(pdb_id, chain, native, stem):
    """canonical PDB 残基序号 → 序列位置（与 native 长度对齐）"""
    reskeys = []
    with open(os.path.join(CANON, f"{stem}.pdb")) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            rn = line[17:20].strip()
            key = (rn, line[22:26].strip(), line[26].strip())
            if key not in reskeys:
                reskeys.append(key)
    seq_keys = list(reskeys)  # canonical 文件只含核苷酸残基（51 号构建时已过滤配体/非映射残基）
    if len(seq_keys) != len(native):
        return None
    ep = os.path.join(EI, f"{pdb_id}.json")
    if not os.path.exists(ep):
        return None
    e = json.load(open(ep))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    return set(i for i, (rn, rn_num, ic) in enumerate(seq_keys) if (rn_num, ic) in iface_keys)


def rec_pair(pool_seqs, native, pos, skip_dash=False):
    """池内 best 界面/整体恢复；'N' 位跳过"""
    npos = set(i for i, c in enumerate(native) if c != "N")
    ipos = sorted(pos & npos)
    if not ipos:
        return None, None
    best_if, best_ov = -1, -1
    for s in pool_seqs:
        if len(s) != len(native):
            continue
        opos = [i for i in npos if not skip_dash or s[i] != '-']
        if not opos:
            continue
        ip = [i for i in ipos if not skip_dash or s[i] != '-']
        if not ip:
            continue
        ov = sum(1 for i in opos if s[i] == native[i]) / len(opos)
        iv = sum(1 for i in ip if s[i] == native[i]) / len(ip)
        best_ov = max(best_ov, ov)
        best_if = max(best_if, iv)
    return (best_if if best_if >= 0 else None), (best_ov if best_ov >= 0 else None)


def nampnn_valid_pos(stem, native):
    """缺坐标链：复现解析得到有效位"""
    os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
    sys.path.insert(0, os.getcwd())
    from data_utils import parse_PDB
    restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
                'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
                'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
    r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
           'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
           'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
           'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
    alph = [r31[r] for r in restypes]
    RM = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}
    d, _, _, _, _ = parse_PDB(os.path.join(CANON, f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    parsed = "".join(RM.get(alph[AA], "?") for AA in d["S"].cpu().numpy())  # 修复：变量名 alph（原 alphabet 未定义）
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
    return pos, len(parsed)


def main():
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    per_chain = {}
    for stem in test:
        native = canon[stem]["seq"]
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native, stem)
        if not pos:
            continue
        entry = {}
        for tag, dname, temps in [("gRNAde", "grnade_out_btest_v2", ("0.1", "0.5", "1.0")),
                                  ("RhoDesign", "rhodesign_out_btest_v2", ("0.1", "0.5", "1.0"))]:
            pool = []
            for t in temps:
                fp = os.path.join(SUR, dname, f"{stem}.{t}.seqs")
                if os.path.exists(fp):
                    pool += [l.strip() for l in open(fp) if l.strip()]
            iv, ov = rec_pair(pool, native, pos)
            if iv is not None:
                entry[tag] = {"iface": iv, "overall": ov, "pool": "3temp"}
        # NA-MPNN
        fp = os.path.join(SUR, "nampnn_out_btest_v2", f"{stem}.seqs")
        if os.path.exists(fp):
            seqs = [l.strip() for l in open(fp) if l.strip()]
            lens = {len(s) for s in seqs}
            if lens == {len(native)}:
                iv, ov = rec_pair(seqs, native, pos)
            else:
                vpos, plen = nampnn_valid_pos(stem, native)
                sub = [s for s in seqs if len(s) == len(vpos)]  # 修复：按有效位长匹配（plen 含 '?' 未知位，5NFV_B 类链会空集）
                npos = [i for i in vpos if native[i] != "N"]
                ipos = sorted(set(pos) & set(npos))
                if sub and ipos:
                    idx = {i: k for k, i in enumerate(vpos)}
                    iv = max(sum(1 for i in ipos if s[idx[i]] == native[i]) / len(ipos) for s in sub)
                    ov = max(sum(1 for i in npos if s[idx[i]] == native[i]) / len(npos) for s in sub)
                else:
                    iv = ov = None
            if iv is not None:
                entry["NAMPNN"] = {"iface": iv, "overall": ov, "pool": "t0.1x10"}
        # AlignIF
        faf = os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta")
        if os.path.exists(faf):
            lines = [l.strip() for l in open(faf) if l.strip()]
            seqs = [lines[j + 1] for j, l in enumerate(lines)
                    if l.startswith(">id:") and j + 1 < len(lines)
                    and all(c in "ACGU-" for c in lines[j + 1])]
            iv, ov = rec_pair(seqs, native, pos, skip_dash=True)
            if iv is not None:
                entry["AlignIF"] = {"iface": iv, "overall": ov, "pool": "t0.1x20"}
        if entry:
            per_chain[stem] = entry

    summary = {}
    for m in ("gRNAde", "RhoDesign", "NAMPNN", "AlignIF"):
        vals = [v[m] for v in per_chain.values() if m in v]
        if vals:
            summary[m] = {
                "n_chains": len(vals),
                "interface_recovery_best": float(np.mean([v["iface"] for v in vals])),
                "overall_recovery_best": float(np.mean([v["overall"] for v in vals])),
            }
    json.dump({"summary": summary, "per_chain": per_chain,
               "note": "B 层 v2 test × canonical native；配体 5.0Å 界面；池内 best 口径"},
              open(os.path.join(SUR, "interface_metrics_btest_v2.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print(f"界面链总数: {len(per_chain)}", flush=True)


if __name__ == "__main__":
    main()
