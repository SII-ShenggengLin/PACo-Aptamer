#!/usr/bin/env python3
"""第 61 步：NA-MPNN B 层 v2 test 指标（canonical native 为唯一真值）

- native = survey/canonical_v2.json seq（含 'N' 未知碱基位，评估跳过）
- 输出长度 < native 的链：复现 NA-MPNN parse_PDB 得到有效位（缺 C1' 坐标残基被官方解析丢弃，
  无位置标记），仅在有效位上评（与 A 层 48 号同口径）
- 口径：t=0.1 × 10 池内 best + t=0.1 均值
输出：survey/metrics_NAMPNN_btest_v2.json
"""
import os, sys, json, difflib
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
NP_OUT = os.path.join(DATA_DIR, "survey", "nampnn_out_btest_v2")
CANON_DIR = os.path.join(DATA_DIR, "canonical_v2")

restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
            'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
            'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
       'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
alphabet = [r31[r] for r in restypes]
RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}


def parse_native(stem):
    os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
    sys.path.insert(0, os.getcwd())
    from data_utils import parse_PDB
    d, _, _, _, _ = parse_PDB(os.path.join(CANON_DIR, f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    S = d["S"].cpu().numpy()
    return "".join(RNA_MAP.get(alphabet[AA], "?") for AA in S)


def align_positions(parsed, native):
    # 修复（2026-08-16）：parsed 中未知 token 位（'?'，如 5NFV_B 首端修饰残基）不可能
    # 进入 equal 块，原断言"equal 拼接==parsed 全长"必然失败 → 改为只校验 equal 块一致，
    # 返回值即有效 native 位置（与 48 号"有效位口径"一致）
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos, ppos = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
            ppos.extend(range(j1, j2))
    assert "".join(native[i] for i in pos) == "".join(parsed[j] for j in ppos), "对齐失败"
    return pos


def main():
    canon = json.load(open(os.path.join(DATA_DIR, "survey", "canonical_v2.json")))
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    results = {}
    gap_chains = {}
    for stem in test:
        out_f = os.path.join(NP_OUT, f"{stem}.seqs")
        if not os.path.exists(out_f):
            continue
        native = canon[stem]["seq"]
        npos = [i for i, c in enumerate(native) if c != "N"]  # 跳过未知碱基位
        seqs = [l.strip() for l in open(out_f) if l.strip()]
        full = [s for s in seqs if len(s) == len(native)]
        if full:
            recs = [sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in full]
            used = len(full)
        else:
            parsed = parse_native(stem)
            pos_all = align_positions(parsed, native)  # 输出位 ↔ native 位（按序）
            sub = [s for s in seqs if len(s) == len(pos_all)]  # 修复：按有效位长匹配（原 len(parsed) 对含 '?' 位链必然空集）
            if not sub or not pos_all:
                continue
            pos = [i for i in pos_all if native[i] != "N"]
            idx = {i: k for k, i in enumerate(pos_all)}
            recs = [sum(1 for i in pos if s[idx[i]] == native[i]) / len(pos) for s in sub]
            used = len(sub)
            gap_chains[stem] = len(pos)
        results[stem] = {
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": float(np.mean(recs)),
            "n_samples": used,
        }
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(test)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])),
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])),
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])),
        "note": "canonical native 真值；'N' 未知碱基位跳过；缺 C1' 坐标链按有效位评估（gap_chains）",
        "gap_chains": gap_chains,
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_NAMPNN_btest_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
