#!/usr/bin/env python3
"""第 34 步：AlignIF 指标计算（恢复率 + 界面口袋恢复率）

口径与 18/28/32 号一致（10 号报告统一口径）：
- 池内 best 恢复率：t=0.1 x 20 样本池内最高
- t=0.1 均值
- 界面恢复率：配体 5.0A 口袋（t=0.1 池内 best）
- gap（'-'，无坐标残基占位）位置跳过：分母 = 非 gap 位置数
输入：survey/alignif_out/{stem}/out.fasta（33 号脚本）
      survey/grnade_out/{stem}.native + eval_inputs + rcsb_pdb（同 32 号）
输出：survey/metrics_AlignIF_v2.json + interface_metrics 并入
"""
import os, json
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
AI_OUT = os.path.join(DATA_DIR, "survey", "alignif_out_v2")
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out_v2")
EI = os.path.join(DATA_DIR, "survey", "eval_inputs_v2")

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

def recovery_skip_gap(seq, native):
    """恢复率：'-'（无坐标残基占位）位置跳过，分母 = 非 gap 位。
    注意：'-' 来自 graph 过滤（C4' 无坐标），native 该位有碱基；
    跳号 '_' 占位已在推理输出层删除（与 native 长度一致）。"""
    hit = tot = 0
    for a, b in zip(seq, native):
        if a == '-':
            continue
        tot += 1
        hit += (a == b)
    return hit / tot if tot else None

def interface_positions(pdb_id, chain, native):
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

def main():
    stems = sorted(d for d in os.listdir(AI_OUT)
                   if os.path.exists(os.path.join(AI_OUT, d, "out.fasta")))
    print(f"计算 {len(stems)} 条链", flush=True)
    results = {}
    iface = defaultdict(dict)
    for stem in stems:
        native_path = os.path.join(GR_OUT, f"{stem}.native")
        if not os.path.exists(native_path):
            continue
        native = open(native_path).read().strip()
        lines = [l.strip() for l in open(os.path.join(AI_OUT, stem, "out.fasta")) if l.strip()]
        seqs = []
        for j, line in enumerate(lines):
            if line.startswith(">id:") and j + 1 < len(lines):
                s = lines[j + 1]
                if len(s) == len(native) and all(c in "ACGU-" for c in s):
                    seqs.append(s)
        if not seqs:
            continue
        recs = [recovery_skip_gap(s, native) for s in seqs]
        recs = [r for r in recs if r is not None]
        if not recs:
            continue
        r = {
            "stem": stem,
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": sum(recs) / len(recs),
            "n_samples": len(seqs),
        }
        results[stem] = r
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native)
        if pos:
            best_iface = -1
            best_overall = -1
            for s in seqs:
                iface_hits = sum(1 for i in pos if s[i] != '-' and s[i] == native[i])
                iface_den = len(pos)
                if iface_den:
                    best_iface = max(best_iface, iface_hits / iface_den)
                rec = recovery_skip_gap(s, native)
                if rec is not None:
                    best_overall = max(best_overall, rec)
            if best_iface >= 0:
                iface[stem] = {"AlignIF": {"iface_t0.1": best_iface, "overall_t0.1": best_overall}}

    recs_best = [r["recovery_best"] for r in results.values()]
    recs_mean = [r["recovery_t01_mean"] for r in results.values()]
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/290",
        "recovery_best_mean": sum(recs_best) / len(recs_best) if recs_best else None,
        "recovery_best_median": sorted(recs_best)[len(recs_best) // 2] if recs_best else None,
        "recovery_t01_mean_mean": sum(recs_mean) / len(recs_mean) if recs_mean else None,
    }
    if iface:
        ifs = [v["AlignIF"]["iface_t0.1"] for v in iface.values()]
        ovs = [v["AlignIF"]["overall_t0.1"] for v in iface.values()]
        summary["interface_recovery_t01_best"] = sum(ifs) / len(ifs)
        summary["overall_recovery_t01_best"] = sum(ovs) / len(ovs)
        summary["n_iface_chains"] = len(ifs)
        im_path = os.path.join(DATA_DIR, "survey", "interface_metrics_v2.json")
        if os.path.exists(im_path):
            im = json.load(open(im_path))
            im["per_chain"] = {**im.get("per_chain", {}), **iface}
            im.setdefault("summary", {})["AlignIF"] = {
                "n_chains": len(ifs),
                "interface_recovery_t01_best": summary["interface_recovery_t01_best"],
                "overall_recovery_t01_best": summary["overall_recovery_t01_best"],
            }
            json.dump(im, open(im_path, "w"), indent=1, ensure_ascii=False)
            print(f"interface_metrics_v2.json 已并入 AlignIF 字段", flush=True)

    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_AlignIF_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== AlignIF 指标（官方权重 x 评估集，use_msta=False） =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    print(f"写入 metrics_AlignIF_v2.json", flush=True)

if __name__ == "__main__":
    main()
