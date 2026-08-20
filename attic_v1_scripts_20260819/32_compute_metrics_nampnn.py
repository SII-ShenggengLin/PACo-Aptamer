#!/usr/bin/env python3
"""第 32 步：NA-MPNN 指标计算（恢复率 + 界面口袋恢复率）

口径与 18/28 号脚本一致（10 号报告统一口径）：
- 池内 best 恢复率：t=0.1 × 10 样本池内最高（NA-MPNN 仅单温度）
- t=0.1 均值：池内平均
- 界面恢复率：配体重原子 5.0Å 内 RNA 残基位置（t=0.1 池内 best）
输入：survey/nampnn_out/{stem}.seqs（每行一条设计序列，30/31 号脚本）
      survey/grnade_out/{stem}.native（native 序列，与 gRNAde 同源）
      survey/eval_inputs/*.json（界面标注）+ rcsb_pdb/{pdb}.pdb（残基序号重建）
输出：survey/metrics_NAMPNN.json + interface_metrics 并入（NA-MPNN 字段）
"""
import os, json
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
NP_OUT = os.path.join(DATA_DIR, "survey", "nampnn_out")
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out")
EI = os.path.join(DATA_DIR, "survey", "eval_inputs")

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

def main():
    stems = sorted(f[:-5] for f in os.listdir(NP_OUT) if f.endswith(".seqs"))
    print(f"计算 {len(stems)} 条链", flush=True)
    results = {}
    iface = defaultdict(dict)
    n_ok = 0
    for stem in stems:
        native_path = os.path.join(GR_OUT, f"{stem}.native")
        if not os.path.exists(native_path):
            continue
        with open(native_path) as f:
            native = f.read().strip()
        seqs = [l.strip() for l in open(os.path.join(NP_OUT, f"{stem}.seqs")) if l.strip()]
        seqs = [s for s in seqs if len(s) == len(native)]
        if not seqs:
            continue
        recs = [recovery(s, native) for s in seqs]
        r = {
            "stem": stem,
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": sum(recs) / len(recs),
            "n_samples": len(seqs),
        }
        results[stem] = r
        n_ok += 1
        # 界面恢复率（28 号口径：t=0.1 池内 best）
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native)
        if pos:
            best_iface = max(sum(1 for i in pos if s[i] == native[i]) / len(pos) for s in seqs)
            best_overall = max(recovery(s, native) for s in seqs)
            iface[stem] = {"NAMPNN": {"iface_t0.1": best_iface, "overall_t0.1": best_overall}}

    recs_best = [r["recovery_best"] for r in results.values()]
    recs_mean = [r["recovery_t01_mean"] for r in results.values()]
    summary = {
        "n_chains": len(results),
        "coverage": f"{n_ok}/290",
        "recovery_best_mean": sum(recs_best) / len(recs_best) if recs_best else None,
        "recovery_best_median": sorted(recs_best)[len(recs_best) // 2] if recs_best else None,
        "recovery_t01_mean_mean": sum(recs_mean) / len(recs_mean) if recs_mean else None,
    }
    # 界面汇总（并入 interface_metrics.json 的 NA-MPNN 字段）
    if iface:
        ifs = [v["NAMPNN"]["iface_t0.1"] for v in iface.values()]
        ovs = [v["NAMPNN"]["overall_t0.1"] for v in iface.values()]
        summary["interface_recovery_t01_best"] = sum(ifs) / len(ifs)
        summary["overall_recovery_t01_best"] = sum(ovs) / len(ovs)
        summary["n_iface_chains"] = len(ifs)
        im_path = os.path.join(DATA_DIR, "survey", "interface_metrics.json")
        if os.path.exists(im_path):
            im = json.load(open(im_path))
            im["per_chain"] = {**im.get("per_chain", {}), **iface}
            im["summary"]["NAMPNN"] = {
                "n_chains": len(ifs),
                "interface_recovery_t01_best": summary["interface_recovery_t01_best"],
                "overall_recovery_t01_best": summary["overall_recovery_t01_best"],
            }
            json.dump(im, open(im_path, "w"), indent=1, ensure_ascii=False)
            print(f"interface_metrics.json 已并入 NA-MPNN 字段", flush=True)

    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_NAMPNN.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n===== NA-MPNN 指标（官方权重 × 评估集） =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    print(f"写入 metrics_NAMPNN.json", flush=True)

if __name__ == "__main__":
    main()
