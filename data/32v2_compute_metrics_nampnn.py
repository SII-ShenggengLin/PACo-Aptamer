#!/usr/bin/env python3
"""第 32 步：NA-MPNN 指标计算（恢复率 + 界面口袋恢复率）

口径与 18/28 号脚本一致（10 号报告统一口径）：
- 池内 best 恢复率：t=0.1 × 10 样本池内最高（NA-MPNN 仅单温度）
- t=0.1 均值：池内平均
- 界面恢复率：配体重原子 5.0Å 内 RNA 残基位置（t=0.1 池内 best）
输入：survey/nampnn_out/{stem}.seqs（每行一条设计序列，30/31 号脚本）
      survey/grnade_out/{stem}.native（native 序列，与 gRNAde 同源）
      survey/eval_inputs/*.json（界面标注）+ rcsb_pdb/{pdb}.pdb（残基序号重建）
输出：survey/metrics_NAMPNN_v2.json（缺 C1' 链按有效位评估,2026-08-19 内联原 48 号补丁）
"""
import os, json
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
NP_OUT = os.path.join(DATA_DIR, "survey", "nampnn_out_v2")
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out_v2")
EI = os.path.join(DATA_DIR, "survey", "eval_inputs_v2")
PDB_IN = os.path.join(DATA_DIR, "survey", "grnade_inputs_v2")

MOD_MAP = {  # 与 28 号脚本相同（修饰核苷酸 → 母碱基）
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

def recovery(seq, native):
    return sum(1 for a, b in zip(seq, native) if a == b) / max(1, len(native))

def interface_positions(pdb_id, chain, native):
    """28 号脚本同款：PDB 残基序号 → 归一化序列位置，返回配体接触位置"""
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

def nampnn_valid_positions(stem, native):
    """NA-MPNN 官方 parse_PDB 以 C1' 建图,原始 PDB 缺 C1' 的残基被丢弃 →
    输出短于 native。复现解析 + difflib 对齐得有效位(原 48 号补丁逻辑,2026-08-19 内联,
    避免"单独重跑 32v2 会覆盖补丁产物"的回归坑)。"""
    import difflib
    os.chdir(os.path.join(os.path.dirname(DATA_DIR), 'tools', 'NA-MPNN', 'inference'))
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
    d, _, _, _, _ = parse_PDB(os.path.join(PDB_IN, f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    parsed = "".join(RM.get(alph[AA], "?") for AA in d["S"].cpu().numpy())
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
    return pos, parsed


def main():
    stems = sorted(f[:-5] for f in os.listdir(NP_OUT) if f.endswith(".seqs"))
    print(f"计算 {len(stems)} 条链", flush=True)
    results = {}
    n_ok = 0
    n_gapfix = 0
    for stem in stems:
        native_path = os.path.join(GR_OUT, f"{stem}.native")
        if not os.path.exists(native_path):
            continue
        with open(native_path) as f:
            native = f.read().strip()
        seqs = [l.strip() for l in open(os.path.join(NP_OUT, f"{stem}.seqs")) if l.strip()]
        full = [s for s in seqs if len(s) == len(native)]
        if full:
            recs = [recovery(s, native) for s in full]
            r = {
                "stem": stem,
                "len": len(native),
                "recovery_best": max(recs),
                "recovery_t01_mean": sum(recs) / len(recs),
                "n_samples": len(full),
            }
        else:
            # 缺 C1' 链:仅在 NA-MPNN 实际解析到的有效位上评估(AlignIF gap 同口径)
            pos, parsed = nampnn_valid_positions(stem, native)
            sub = [s for s in seqs if len(s) == len(parsed)]
            if not sub:
                continue
            idx = {i: k for k, i in enumerate(pos)}
            recs = [sum(1 for i in pos if s[idx[i]] == native[i]) / len(pos) for s in sub]
            r = {
                "stem": stem,
                "len": len(native),
                "recovery_best": max(recs),
                "recovery_t01_mean": sum(recs) / len(recs),
                "n_samples": len(sub),
                "n_valid_pos": len(pos),
            }
            n_gapfix += 1
        results[stem] = r
        n_ok += 1

    import numpy as np
    recs_best = [r["recovery_best"] for r in results.values()]
    recs_mean = [r["recovery_t01_mean"] for r in results.values()]
    summary = {
        "n_chains": len(results),
        "coverage": f"{n_ok}/289",
        "recovery_best_mean": float(np.mean(recs_best)) if recs_best else None,
        "recovery_best_median": float(np.median(recs_best)) if recs_best else None,
        "recovery_t01_mean_mean": float(np.mean(recs_mean)) if recs_mean else None,
    }
    if n_gapfix:
        summary["gap_fix_note"] = (f"{n_gapfix} 链原始 PDB 缺 C1' 坐标,NA-MPNN 解析丢弃这些残基;"
                                   "按 AlignIF gap 口径仅在有效位上评估")
    # 界面口径已由 74/77 号逐残基重算链路取代,本脚本不再写 interface 文件(2026-08-19)

    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_NAMPNN_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== NA-MPNN 指标（官方权重 × 评估集） =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    print(f"写入 metrics_NAMPNN_v2.json", flush=True)

if __name__ == "__main__":
    main()
