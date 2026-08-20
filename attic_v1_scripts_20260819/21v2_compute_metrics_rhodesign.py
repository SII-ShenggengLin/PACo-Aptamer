#!/usr/bin/env python3
"""第 21v2 步：RhoDesign v2 指标（恢复率 + SS 一致性 + 界面口袋恢复率）

v2 适配：rhodesign_out_v2（新划分 A 层 289 链）、grnade_out_v2（native）、
grnade_inputs_v2（SS）、eval_inputs_v2（界面标注）；界面并入 interface_metrics_v2.json。
口径与 32v2/34v2 一致（10 号报告统一口径：池内 best / t=0.1 均值）。
"""
import os, re, json, subprocess
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
GR_OUT = os.path.join(DATA_DIR, "survey", "rhodesign_out_v2")
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs_v2")
EI = os.path.join(DATA_DIR, "survey", "eval_inputs_v2")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
TEMPS = [0.1, 0.5, 1.0]

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

def recovery(seq, native):
    return sum(1 for a, b in zip(seq, native) if a == b) / max(1, len(native))

def bracket_from_rnafold(stdout):
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", stdout)
    return m[0] if m else None

def ss_f1(seq, native_ss):
    """设计序列的 RNAfold SS vs 天然序列 RNAfold SS 的配对 F1"""
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n", capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    b = bracket_from_rnafold(r.stdout)
    if not b or len(b) != len(native_ss):
        return None
    n = len(b)
    tp = sum(1 for i in range(n) for j in range(i+1, n)
             if native_ss[i] != "." and native_ss[j] != "." and b[i] == "(" and b[j] == ")"
             and native_ss[i] == "(" and native_ss[j] == ")")
    # 简化：用括号位置匹配（pair F1 的 O(n²) 精确版）
    nat_pairs = set()
    stack = []
    for i, c in enumerate(native_ss):
        if c in "([{<":
            stack.append((i, c))
        elif c in ")]}>":
            if stack:
                nat_pairs.add((stack.pop()[0], i))
    pred_pairs = set()
    stack = []
    for i, c in enumerate(b):
        if c in "([{<":
            stack.append((i, c))
        elif c in ")]}>":
            if stack:
                pred_pairs.add((stack.pop()[0], i))
    inter = len(nat_pairs & pred_pairs)
    if not nat_pairs:
        return None
    prec = inter / max(1, len(pred_pairs))
    rec_ = inter / len(nat_pairs)
    return 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0

def work_one(stem):
    """计算单链全部指标"""
    native_path = os.path.join(GR_OUT, f"{stem}.native")
    if not os.path.exists(native_path):
        return (stem, None)
    with open(native_path) as f:
        native = f.read().strip()
    native_ss = None
    ss_path = os.path.join(IN_DIR, f"{stem}.ss")
    if os.path.exists(ss_path):
        native_ss = open(ss_path).read().strip()

    result = {"stem": stem, "len": len(native)}
    all_recs = []
    for t in TEMPS:
        sp = os.path.join(GR_OUT, f"{stem}.{t}.seqs")
        if not os.path.exists(sp):
            continue
        seqs = [l.strip() for l in open(sp) if l.strip()]
        recs = [recovery(s, native) for s in seqs]
        all_recs.extend(recs)
        result[f"recovery_t{t}"] = {
            "best": max(recs) if recs else None,
            "mean": sum(recs) / len(recs) if recs else None,
            "n": len(recs),
        }
    # best 恢复率样本的 SS F1
    best_seq = None
    best_r = -1
    for t in TEMPS:
        sp = os.path.join(GR_OUT, f"{stem}.{t}.seqs")
        if not os.path.exists(sp):
            continue
        for s in (l.strip() for l in open(sp) if l.strip()):
            r = recovery(s, native)
            if r > best_r:
                best_r, best_seq = r, s
    result["recovery_best_overall"] = best_r
    if native_ss and best_seq:
        result["ss_f1_best"] = ss_f1(best_seq, native_ss)
    # 池内样本多样性（唯一序列比例）
    all_seqs = set()
    for t in TEMPS:
        sp = os.path.join(GR_OUT, f"{stem}.{t}.seqs")
        if os.path.exists(sp):
            all_seqs.update(l.strip() for l in open(sp) if l.strip())
    result["unique_ratio"] = len(all_seqs) / max(1, len(all_recs))
    return (stem, result)

def main():
    stems = sorted(f[:-7] for f in os.listdir(GR_OUT) if f.endswith(".native"))
    print(f"计算 {len(stems)} 条链", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, r) in enumerate(ex.map(work_one, stems)):
            if r:
                results[stem] = r
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{len(stems)}", flush=True)

    # 汇总
    recs_best = [r["recovery_best_overall"] for r in results.values() if r.get("recovery_best_overall") is not None]
    recs_t01 = [r["recovery_t0.1"]["best"] for r in results.values() if r.get("recovery_t0.1", {}).get("best")]
    ss_f1s = [r["ss_f1_best"] for r in results.values() if r.get("ss_f1_best") is not None]
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/289",
        "recovery_best_mean": sum(recs_best) / len(recs_best) if recs_best else None,
        "recovery_best_median": sorted(recs_best)[len(recs_best)//2] if recs_best else None,
        "recovery_t01_best_mean": sum(recs_t01) / len(recs_t01) if recs_t01 else None,
        "ss_f1_best_mean": sum(ss_f1s) / len(ss_f1s) if ss_f1s else None,
        "n_with_ss": len(ss_f1s),
    }

    # 界面口袋恢复率（3 温度全池内 best，配体 5.0Å 口袋）——并入 interface_metrics_v2.json
    # 口径与旧 28 号一致（RhoDesign/gRNAde 300 池；28v2 同款）；32v2/34v2 单温度池为方法输出所限
    iface = defaultdict(dict)
    for stem, r in results.items():
        native_path = os.path.join(GR_OUT, f"{stem}.native")
        if not os.path.exists(native_path):
            continue
        native = open(native_path).read().strip()
        seqs = []
        for t in TEMPS:
            sp = os.path.join(GR_OUT, f"{stem}.{t}.seqs")
            if os.path.exists(sp):
                seqs.extend(l.strip() for l in open(sp) if l.strip())
        if not seqs:
            continue
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native)
        if pos:
            best_iface = max(sum(1 for i in pos if s[i] == native[i]) / len(pos) for s in seqs)
            best_overall = max(recovery(s, native) for s in seqs)
            iface[stem] = {"RhoDesign": {"iface_t0.1": best_iface, "overall_t0.1": best_overall}}
    if iface:
        ifs = [v["RhoDesign"]["iface_t0.1"] for v in iface.values()]
        ovs = [v["RhoDesign"]["overall_t0.1"] for v in iface.values()]
        summary["interface_recovery_t01_best"] = sum(ifs) / len(ifs)
        summary["overall_recovery_t01_best"] = sum(ovs) / len(ovs)
        summary["n_iface_chains"] = len(ifs)
        im_path = os.path.join(DATA_DIR, "survey", "interface_metrics_v2.json")
        if os.path.exists(im_path):
            im = json.load(open(im_path))
            im["per_chain"] = {**im.get("per_chain", {}), **iface}
            im.setdefault("summary", {})["RhoDesign"] = {
                "n_chains": len(ifs),
                "interface_recovery_t01_best": summary["interface_recovery_t01_best"],
                "overall_recovery_t01_best": summary["overall_recovery_t01_best"],
            }
            json.dump(im, open(im_path, "w"), indent=1, ensure_ascii=False)
            print(f"interface_metrics_v2.json 已并入 RhoDesign 字段", flush=True)

    out = {"summary": summary, "per_chain": results}
    json.dump(out, open(os.path.join(DATA_DIR, "survey", "metrics_RhoDesign_v2.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n===== RhoDesign 指标（official 权重 × 评估集 v2） =====", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"写入 metrics_RhoDesign_v2.json", flush=True)

if __name__ == "__main__":
    main()
