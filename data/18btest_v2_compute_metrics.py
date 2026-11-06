#!/usr/bin/env python3
"""第 18 步：对比方法指标计算（恢复率 + SS 一致性）

输入：survey/grnade_out/{stem}.{temp}.seqs + .native（gRNAde 采样）
      survey/grnade_inputs/{stem}.ss（天然序列的 RNAfold SS，16 号脚本）
输出：survey/metrics_{method}.json

指标定义（09 基准方案第三节）：
1. 恢复率：同位置 4 字母匹配；best = 池内最高（多序列采样）；mean = 池平均
2. SS F1：RNAfold 预测设计序列 SS vs RNAfold 预测天然序列 SS 的配对 F1
   （同一预测器，避免天然 SS 口径偏差；dot-bracket 归一化）
"""
import os, re, json, subprocess
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out_btest_v2")
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs_btest_v2")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
TEMPS = [0.1, 0.5, 1.0]

def recovery(seq, native):
    """同位点匹配率；native 的 N 位不计分母（canonical 口径，2026-08-19 对齐 63 号；
    曾把 N 计入分母 → 7 条含 N 链恢复率被硬性压顶）"""
    pos = [i for i, b in enumerate(native) if b != "N"]
    if not pos:
        return None
    return sum(1 for i in pos if i < len(seq) and seq[i] == native[i]) / len(pos)

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
        recs = [r for r in (recovery(s, native) for s in seqs) if r is not None]
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
            if r is not None and r > best_r:
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
    import numpy as np
    recs_best = [r["recovery_best_overall"] for r in results.values() if r.get("recovery_best_overall") is not None]
    recs_t01 = [r["recovery_t0.1"]["best"] for r in results.values()
                if r.get("recovery_t0.1", {}).get("best") is not None]
    ss_f1s = [r["ss_f1_best"] for r in results.values() if r.get("ss_f1_best") is not None]
    summary = {
        "n_chains": len(results),
        "recovery_best_mean": float(np.mean(recs_best)) if recs_best else None,
        "recovery_best_median": float(np.median(recs_best)) if recs_best else None,
        "recovery_t01_best_mean": float(np.mean(recs_t01)) if recs_t01 else None,
        "ss_f1_best_mean": float(np.mean(ss_f1s)) if ss_f1s else None,
        "n_with_ss": len(ss_f1s),
    }
    out = {"summary": summary, "per_chain": results}
    json.dump(out, open(os.path.join(DATA_DIR, "survey", "metrics_gRNAde_btest_v2.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n===== gRNAde 指标（B 层 v2 重训 × test 477） =====", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"写入 metrics_gRNAde_btest_v2.json", flush=True)

if __name__ == "__main__":
    main()
