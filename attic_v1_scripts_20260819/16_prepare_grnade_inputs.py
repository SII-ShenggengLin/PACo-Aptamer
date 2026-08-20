#!/usr/bin/env python3
"""第 16 步：为 gRNAde 准备评估输入（链级 PDB + 二级结构 dot-bracket）

1. 从 rcsb_pdb 切出 eval 条目的 RNA 链为独立 PDB（链重命名 A，对齐 03 脚本口径）
2. RNAfold 预测二级结构（viennarna_fold_env 的 RNAfold 2.7.2，评估输入不含任何预测器）
输出：survey/grnade_inputs/{pdb}_{chain}.pdb + .ss（dot-bracket）
"""
import os, subprocess, json, re
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
OUT_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs")
os.makedirs(OUT_DIR, exist_ok=True)
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"

# 与 14 号脚本一致的修饰映射（简版：切链时只保留母碱基残基名不变，RNAfold 需要纯 A/C/G/U）
STD = {"A", "C", "G", "U"}

def split_chain(pdb_id, chain, out_stem):
    """从条目 PDB 切出单链（ATOM + 该链 HETATM 修饰），链重命名 A"""
    src = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    lines_out = []
    seq = []
    with open(src) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM") and line[21] == chain and line[16] in (" ", "A"):
                rn = line[17:20].strip()
                if rn not in STD:
                    continue   # 修饰残基丢弃（母碱基在 ATOM 记录里通常也有）
                new = line[:21] + "A" + line[22:]
                lines_out.append(new)
            elif rec == "TER" and line[21] == chain:
                lines_out.append(line)
            elif rec == "END":
                lines_out.append(line)
    if not lines_out:
        return None
    pdb_out = out_stem + ".pdb"
    with open(pdb_out, "w") as f:
        f.writelines(lines_out)
    # 提取序列（按 ATOM 顺序去重残基）
    seen = set()
    for line in lines_out:
        if line[:6].strip() == "ATOM":
            rn = line[17:20].strip()
            seqnum = line[22:26].strip()
            if (rn, seqnum) not in seen:
                seen.add((rn, seqnum))
                seq.append(rn)
    return "".join(seq)

def predict_ss(seq, out_stem):
    """RNAfold 预测二级结构 → dot-bracket 字符串"""
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n", capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    # 输出第 3 行是 dot-bracket（可多行续接）
    lines = r.stdout.strip().split("\n")
    ss = ""
    for line in lines[1:]:
        ss += line.split()[0] if line.split() else ""
        if "(" in ss or ")" in ss or "." in ss:
            # 收完所有连续行
            pass
    # RNAfold 输出：第一行序列，后续行带能量。取能量行之前的括号串
    bracket = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return bracket[0] if bracket else None

def work(args):
    pdb_id, chain = args
    out_stem = os.path.join(OUT_DIR, f"{pdb_id}_{chain}")
    if os.path.exists(out_stem + ".pdb") and os.path.exists(out_stem + ".ss"):
        return (pdb_id, chain, "cached")
    seq = split_chain(pdb_id, chain, out_stem)
    if not seq:
        return (pdb_id, chain, "no_chain")
    ss = predict_ss(seq, out_stem)
    if not ss:
        return (pdb_id, chain, "no_ss")
    with open(out_stem + ".ss", "w") as f:
        f.write(ss + "\n")
    with open(out_stem + ".seq", "w") as f:
        f.write(seq + "\n")
    return (pdb_id, chain, f"ok_{len(seq)}nt")

def main():
    # 收集 eval 条目链
    tasks = []
    for fn in sorted(os.listdir(os.path.join(DATA_DIR, "survey", "eval_inputs"))):
        if not fn.endswith(".json") or fn == "eval_manifest.json":
            continue
        e = json.load(open(os.path.join(DATA_DIR, "survey", "eval_inputs", fn)))
        for ch in e["sequences"]:
            tasks.append((e["pdb"], ch))
    print(f"待切链: {len(tasks)}", flush=True)
    stats = {"cached": 0, "no_chain": 0, "no_ss": 0}
    ok = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (pdb, ch, status) in enumerate(ex.map(work, tasks)):
            if status.startswith("ok"):
                ok += 1
            else:
                stats[status] = stats.get(status, 0) + 1
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{len(tasks)} ok={ok}", flush=True)
    print(f"完成: ok={ok} {stats}", flush=True)

if __name__ == "__main__":
    main()
