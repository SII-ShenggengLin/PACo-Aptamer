#!/usr/bin/env python3
"""第 28btest_v2 步：B 层 v2 界面口袋恢复率（test 477 链，gRNAde + RhoDesign 统一计算）

界面定义：配体重原子 5.0Å 内 RNA 残基（44 号脚本标注，eval_inputs_v2 覆盖 test 全链）
口径与 28v2（A 层）完全一致：3 温度全池内 best（gRNAde/RhoDesign 各 300 池）。
TACo 无界面字段（38 号评估不保存序列池，如实注明）。
输入：eval_inputs_v2/*.json（界面标注）+ grnade_out_btest_v2 + rhodesign_out_btest_v2
输出：survey/interface_metrics_btest_v2.json（B 层专用，勿覆盖 A 层 interface_metrics_v2.json）
"""
import os, json
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
EI = os.path.join(DATA_DIR, "survey", "eval_inputs_v2")
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out_btest_v2")
RH_OUT = os.path.join(DATA_DIR, "survey", "rhodesign_out_btest_v2")
NP_OUT = os.path.join(DATA_DIR, "survey", "nampnn_out_v2")
AI_OUT = os.path.join(DATA_DIR, "survey", "alignif_out_v2")
TEMPS = [0.1, 0.5, 1.0]

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
STD = {"A", "C", "G", "U"}

METHODS = {
    "gRNAde": GR_OUT,
    "RhoDesign": RH_OUT,
}


def interface_positions(pdb_id, chain, native):
    """PDB 残基序号 → 归一化序列位置映射，返回界面位置（配体接触残基）"""
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
        return set()
    e = json.load(open(os.path.join(EI, f"{pdb_id}.json")))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    positions = set()
    for i, (rn, rn_num, ic) in enumerate(seq_keys):
        if (rn_num, ic) in iface_keys:
            positions.add(i)
    return positions


def seqs_for(method, stem, native):
    """读取方法样本池 → 有效序列列表"""
    if method == "NAMPNN":
        sp = os.path.join(NP_OUT, f"{stem}.seqs")
        if not os.path.exists(sp):
            return []
        return [s for s in (l.strip() for l in open(sp)) if len(s) == len(native)]
    if method == "AlignIF":
        d = os.path.join(AI_OUT, stem)
        if not os.path.exists(os.path.join(d, "out.fasta")):
            return []
        lines = [l.strip() for l in open(os.path.join(d, "out.fasta")) if l.strip()]
        seqs = []
        for j, line in enumerate(lines):
            if line.startswith(">id:") and j + 1 < len(lines):
                s = lines[j + 1]
                if len(s) == len(native) and all(c in "ACGU-" for c in s):
                    seqs.append(s)
        return seqs
    # gRNAde / RhoDesign：3 温度 seqs 文件
    seqs = []
    for t in TEMPS:
        sp = os.path.join(METHODS[method], f"{stem}.{t}.seqs")
        if os.path.exists(sp):
            seqs.extend(l.strip() for l in open(sp) if l.strip())
    return [s for s in seqs if len(s) == len(native)]


def recovery(seq, native):
    return sum(1 for a, b in zip(seq, native) if a == b) / max(1, len(native))


def recovery_skip_gap(seq, native):
    hit = tot = 0
    for a, b in zip(seq, native):
        if a == '-':
            continue
        tot += 1
        hit += (a == b)
    return hit / tot if tot else None


def main():
    results = defaultdict(dict)
    # 从 grnade_out_v2 的 native 出发（与 32v2/34v2 完全同口径）——
    # eval_inputs_v2 的 sequences 与 grnade native 长度有少量差异（如 1MMS_C 57 vs 58），
    # 用 grnade native 保证 len 对齐检查一致。
    for stem in sorted(os.listdir(GR_OUT)):
        if not stem.endswith(".native"):
            continue
        stem = stem[:-7]
        native_path = os.path.join(GR_OUT, f"{stem}.native")
        if not os.path.exists(native_path):
            continue
        native = open(native_path).read().strip()
        pdb, ch = stem.rsplit("_", 1)
        positions = interface_positions(pdb, ch, native)
        if not positions:
            continue
        for method in METHODS:
            seqs = seqs_for(method, stem, native)
            if not seqs:
                continue
            best_iface, best_overall = -1, -1
            for s in seqs:
                if method == "AlignIF":
                    hits = sum(1 for i in positions if s[i] != '-' and s[i] == native[i])
                    iface = hits / len(positions)
                    ov = recovery_skip_gap(s, native)
                else:
                    iface = sum(1 for i in positions if s[i] == native[i]) / len(positions)
                    ov = recovery(s, native)
                best_iface = max(best_iface, iface)
                if ov is not None:
                    best_overall = max(best_overall, ov)
            if best_iface >= 0:
                results[stem].setdefault(method, {})["iface_t0.1"] = best_iface
                results[stem].setdefault(method, {})["overall_t0.1"] = best_overall
    # 汇总（t=0.1 池内 best 口径，与 32v2/34v2 一致）
    summary = {}
    for method in METHODS:
        ifaces = [v[method]["iface_t0.1"] for v in results.values() if method in v]
        overalls = [v[method]["overall_t0.1"] for v in results.values() if method in v]
        summary[method] = {
            "n_chains": len(ifaces),
            "interface_recovery_t01_best": sum(ifaces) / len(ifaces) if ifaces else None,
            "overall_recovery_t01_best": sum(overalls) / len(overalls) if overalls else None,
        }
    json.dump({"summary": summary, "per_chain": {k: dict(v) for k, v in results.items()}},
              open(os.path.join(DATA_DIR, "survey", "interface_metrics_btest_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"===== B 层 v2 界面口袋恢复率（test 477 链，配体 5.0Å 口袋，3 温度池内 best） =====")
    for method, s in summary.items():
        print(f"  {method}: 界面恢复率 {s['interface_recovery_t01_best']:.4f} | 整体 {s['overall_recovery_t01_best']:.4f} | n={s['n_chains']}")


if __name__ == "__main__":
    main()
