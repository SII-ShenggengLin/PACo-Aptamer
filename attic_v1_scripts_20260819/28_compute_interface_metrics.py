#!/usr/bin/env python3
"""第 28 步：界面口袋恢复率（所有方法统一计算）

界面定义：配体重原子 5.0Å 内 RNA 残基（14 号脚本标注）
对每个评估条目-链：
  1. 定位该链的界面残基（配体接触的残基序号）
  2. 界面恢复率 = 设计序列在界面残基位置的匹配率
  3. 整体恢复率（对照）
输入：eval_inputs/*.json（界面标注）+ grnade_out/ rhodesign_out/（方法序列）
输出：survey/interface_metrics.json
"""
import os, json
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
EI = os.path.join(DATA_DIR, "survey", "eval_inputs")
METHODS = {
    "gRNAde": os.path.join(DATA_DIR, "survey", "grnade_out"),
    "RhoDesign": os.path.join(DATA_DIR, "survey", "rhodesign_out"),
}
TEMPS = [0.1, 0.5, 1.0]

# 修饰核苷酸 → 母碱基（与 14 号脚本一致）
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

def interface_positions(pdb_id, chain, native):
    """从 PDB 重建链残基序号 → 归一化序列位置映射，返回界面位置（配体接触残基）"""
    # 残基序号列表（与 eval_inputs 的 to_seq 相同过滤）
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
    # 只保留 RNA 残基（STD + MOD_MAP）
    seq_keys = [k for k in reskeys if k[0] in STD or k[0] in MOD_MAP]
    if len(seq_keys) != len(native):
        return set()
    # 界面残基（来自 eval_inputs 标注）
    e = json.load(open(os.path.join(EI, f"{pdb_id}.json")))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    # 残基序号 → 位置
    positions = set()
    for i, (rn, rn_num, ic) in enumerate(seq_keys):
        if (rn_num, ic) in iface_keys:
            positions.add(i)
    return positions


def recovery_on_positions(seq, native, positions):
    """给定位置的恢复率"""
    if not positions or len(seq) != len(native):
        return None
    hit = sum(1 for i in positions if seq[i] == native[i])
    return hit / len(positions)

def main():
    results = defaultdict(dict)
    for fn in sorted(os.listdir(EI)):
        if not fn.endswith(".json") or fn == "eval_manifest.json":
            continue
        e = json.load(open(os.path.join(EI, fn)))
        pdb = e["pdb"]
        for ch, native in e["sequences"].items():
            # 该链的界面残基（PDB 残基序号 → 归一化序列位置）
            chain_positions = interface_positions(pdb, ch, native)
            if not chain_positions:
                continue
            for method, out_dir in METHODS.items():
                for t in TEMPS:
                    sp = os.path.join(out_dir, f"{pdb}_{ch}.{t}.seqs")
                    if not os.path.exists(sp):
                        continue
                    best_iface, best_overall = -1, -1
                    for line in open(sp):
                        seq = line.strip()
                        if len(seq) != len(native):
                            continue
                        iface = recovery_on_positions(seq, native, chain_positions)
                        if iface is not None:
                            best_iface = max(best_iface, iface)
                        best_overall = max(best_overall,
                                           sum(1 for a, b in zip(seq, native) if a == b) / len(native))
                    if best_iface >= 0:
                        results[f"{pdb}_{ch}"].setdefault(method, {})[f"iface_t{t}"] = best_iface
                        results[f"{pdb}_{ch}"].setdefault(method, {})[f"overall_t{t}"] = best_overall
    # 汇总
    summary = {}
    for method in METHODS:
        ifaces = [v[method]["iface_t0.1"] for v in results.values() if method in v and "iface_t0.1" in v[method]]
        overalls = [v[method]["overall_t0.1"] for v in results.values() if method in v and "overall_t0.1" in v[method]]
        summary[method] = {
            "n_chains": len(ifaces),
            "interface_recovery_t01_best": sum(ifaces) / len(ifaces) if ifaces else None,
            "overall_recovery_t01_best": sum(overalls) / len(overalls) if overalls else None,
        }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "interface_metrics.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"===== 界面口袋恢复率（配体 5.0Å 口袋） =====")
    for method, s in summary.items():
        print(f"  {method}: 界面恢复率 {s['interface_recovery_t01_best']:.3f} | 整体 {s['overall_recovery_t01_best']:.3f} | n={s['n_chains']}")

if __name__ == "__main__":
    main()
