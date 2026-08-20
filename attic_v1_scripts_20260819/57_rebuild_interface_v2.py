#!/usr/bin/env python3
"""第 57 步：A 层 v2 界面指标统一重建（修复 per_chain 深合并 bug）

旧问题：32v2/34v2/28v2 各方法脚本用 `{**old, **new}` 浅合并写 interface_metrics_v2.json，
后跑的方法会把同链其他方法的 per_chain 条目整体覆盖 → 本脚本一次性重算四方法并正确合并。

口径（与 10 号报告十三节一致）：
- 界面位 = 配体重原子 5.0Å 内 RNA 残基（eval_inputs_v2 标注 + rcsb_pdb 残基序号映射，
  要求 rcsb 解析残基数 == native 长度，否则该链不计界面）
- gRNAde/RhoDesign：3 温度全池池内 best（界面与整体同池）
- NA-MPNN：t=0.1×10 池内 best；5 条缺 C1' 坐标链（5NFV_B/7E9I_A/8U5J_A/8VPV_A/9GUT_X）
  仅在 NA-MPNN 实际解析位上评（nampnn_valid_positions_v2.json）
- AlignIF：t=0.1×20 池内 best，'-' gap 位跳过
输出：survey/interface_metrics_v2.json（per_chain 四方法齐全 + summary 重算）
"""
import os, json
import numpy as np

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
SUR = os.path.join(DATA_DIR, "survey")
EI = os.path.join(SUR, "eval_inputs_v2")

import importlib.util
spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}


def interface_positions(pdb_id, chain, native):
    """28 号同款：PDB 残基序号 → 归一化序列位置"""
    reskeys = []
    with open(os.path.join(DATA_DIR, "rcsb_pdb", f"{pdb_id}.pdb")) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[21] != chain or line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            if rec == "HETATM" and rn not in MOD_MAP:
                continue
            key = (rn, line[22:26].strip(), line[26].strip())
            if key not in reskeys:
                reskeys.append(key)
    seq_keys = [k for k in reskeys if k[0] in STD or k[0] in MOD_MAP]
    if len(seq_keys) != len(native):
        return None
    e = json.load(open(os.path.join(EI, f"{pdb_id}.json")))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    return set(i for i, (rn, rn_num, ic) in enumerate(seq_keys) if (rn_num, ic) in iface_keys)


def rec_pair(seqs, native, pos, skip_dash=False, valid_pos=None):
    """返回 (池内best界面恢复, 池内best整体恢复)；整体口径与界面一致（同池 best）"""
    best_if, best_ov = -1, -1
    for s in seqs:
        if valid_pos is not None:
            if len(s) != len(valid_pos):
                continue
            ov = sum(1 for k, i in enumerate(valid_pos) if s[k] == native[i]) / len(valid_pos)
            if pos:
                pv = [i for i in pos if i in valid_pos]
                if not pv:
                    continue
                idx = {i: k for k, i in enumerate(valid_pos)}
                iv = sum(1 for i in pv if s[idx[i]] == native[i]) / len(pv)
            else:
                continue
        else:
            if len(s) != len(native):
                continue
            if skip_dash:
                den = sum(1 for a in s if a != '-')
                ov = sum(1 for a, b in zip(s, native) if a != '-' and a == b) / den if den else None
            else:
                ov = sum(1 for a, b in zip(s, native) if a == b) / len(native)
            if ov is None:
                continue
            if pos:
                den = sum(1 for i in pos if s[i] != '-') if skip_dash else len(pos)
                if not den:
                    continue
                iv = sum(1 for i in pos if s[i] != '-' and s[i] == native[i]) / den
            else:
                continue
        best_ov = max(best_ov, ov)
        best_if = max(best_if, iv)
    return best_if, best_ov


def main():
    vp = {}
    vpp = os.path.join(SUR, "nampnn_valid_positions_v2.json")
    if os.path.exists(vpp):
        vp = json.load(open(vpp))

    # 收集 289 链
    stems = sorted(f[:-7] for f in os.listdir(os.path.join(SUR, "grnade_out_v2")) if f.endswith(".native"))
    per_chain = {}
    for stem in stems:
        native = open(os.path.join(SUR, "grnade_out_v2", f"{stem}.native")).read().strip()
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native)
        if not pos:
            continue
        entry = {}
        # gRNAde：3 温度全池
        pool = []
        for t in ("0.1", "0.5", "1.0"):
            fp = os.path.join(SUR, "grnade_out_v2", f"{stem}.{t}.seqs")
            if os.path.exists(fp):
                pool += [l.strip() for l in open(fp) if l.strip()]
        iv, ov = rec_pair(pool, native, pos)
        if iv >= 0:
            entry["gRNAde"] = {"iface_t0.1": iv, "overall_t0.1": ov, "pool": "3temp"}
        # RhoDesign：3 温度全池
        pool = []
        for t in ("0.1", "0.5", "1.0"):
            fp = os.path.join(SUR, "rhodesign_out_v2", f"{stem}.{t}.seqs")
            if os.path.exists(fp):
                pool += [l.strip() for l in open(fp) if l.strip()]
        iv, ov = rec_pair(pool, native, pos)
        if iv >= 0:
            entry["RhoDesign"] = {"iface_t0.1": iv, "overall_t0.1": ov, "pool": "3temp"}
        # NA-MPNN：t=0.1×10（有效位口径）
        fp = os.path.join(SUR, "nampnn_out_v2", f"{stem}.seqs")
        if os.path.exists(fp):
            seqs = [l.strip() for l in open(fp) if l.strip()]
            if stem in vp:
                iv, ov = rec_pair(seqs, native, pos, valid_pos=set(vp[stem]))
            else:
                iv, ov = rec_pair(seqs, native, pos)
            if iv >= 0:
                entry["NAMPNN"] = {"iface_t0.1": iv, "overall_t0.1": ov, "pool": "t0.1x10"}
        # AlignIF：t=0.1×20 gap-skip
        faf = os.path.join(SUR, "alignif_out_v2", stem, "out.fasta")
        if os.path.exists(faf):
            lines = [l.strip() for l in open(faf) if l.strip()]
            seqs = [lines[j + 1] for j, l in enumerate(lines)
                    if l.startswith(">id:") and j + 1 < len(lines)
                    and len(lines[j + 1]) == len(native)
                    and all(c in "ACGU-" for c in lines[j + 1])]
            iv, ov = rec_pair(seqs, native, pos, skip_dash=True)
            if iv >= 0:
                entry["AlignIF"] = {"iface_t0.1": iv, "overall_t0.1": ov, "pool": "t0.1x20"}
        if entry:
            per_chain[stem] = entry

    summary = {}
    for m in ("gRNAde", "RhoDesign", "NAMPNN", "AlignIF"):
        vals = [v[m] for v in per_chain.values() if m in v]
        if vals:
            summary[m] = {
                "n_chains": len(vals),
                "interface_recovery_t01_best": float(np.mean([v["iface_t0.1"] for v in vals])),
                "overall_recovery_t01_best": float(np.mean([v["overall_t0.1"] for v in vals])),
            }
    json.dump({"summary": summary, "per_chain": per_chain},
              open(os.path.join(SUR, "interface_metrics_v2.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
