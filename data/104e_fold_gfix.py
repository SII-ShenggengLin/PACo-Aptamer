#!/usr/bin/env python3
"""104e：gRNAde _cfix 池 RNAfold 批量回折缓存（供 77_p03 INF 与 82 GC/MFE 重算）

对 test 477 链（s1 缓存口径，len<=512）：
  pool_full = grnade_out_btest_v2_cfix/{stem}.{0.1,0.5,1.0}.seqs 拼接（len==native 过滤）
  uniq = pool_full 中纯 ACGU 序列去重 → 一次 RNAfold --noPS 子进程
  逐序列记录 (dot-bracket, MFE)
  ref_pairs = s1 缓存 ss_pairs（77 号 pairs_from_matrix 同款）
输出：survey/104_fold_cache_grnade_cfix.json（临时文件 + rename）
环境：mpl372_repro（torch 读 s1 缓存）；PYTHONPATH=data/survey/pyshims
"""
import os, sys, json, time, tempfile, subprocess, re
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
CFIX_DIR = os.path.join(SUR, "grnade_out_btest_v2_cfix")
TEMPS = ("0.1", "0.5", "1.0")
OUT = os.path.join(SUR, "104_fold_cache_grnade_cfix.json")


def pairs_from_matrix(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return {(int(i), int(j)) for i, j, v in zip(*iu, mat[iu]) if v}


def fold_batch(seqs):
    """77 号 fold_batch 同款 + 同步解析 MFE"""
    if not seqs:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        f.write("\n".join(seqs) + "\n")
        fn = f.name
    try:
        r = subprocess.run([RNAFOLD, "--noPS"], stdin=open(fn),
                           capture_output=True, text=True, timeout=600)
        struct_lines = [l for l in r.stdout.splitlines() if l and l[0] in ".([{<"]
        out = []
        for l in struct_lines:
            ms = re.match(r"([\.\(\)\[\]\{\}\<\>]+)", l.strip())
            mm = re.search(r"\(\s*(-?\d+\.\d+)\)\s*$", l)
            if ms:
                out.append((ms.group(1), float(mm.group(1)) if mm else None))
        if len(out) != len(seqs):
            print(f"    [fold 数量不匹配 {len(out)}/{len(seqs)}]", flush=True)
            return []
        return out
    finally:
        os.unlink(fn)


def work_one(stem_seq):
    stem, native = stem_seq
    pool = []
    for t in TEMPS:
        fp = os.path.join(CFIX_DIR, f"{stem}.{t}.seqs")
        if os.path.exists(fp):
            pool += [l.strip() for l in open(fp)
                     if l.strip() and len(l.strip()) == len(native)]
    uniq = sorted({s for s in pool if set(s) <= set("ACGU")})
    if not uniq:
        return stem, None
    res = fold_batch(uniq)
    if not res:
        return stem, None
    fmap = {s: [st, mfe] for s, (st, mfe) in zip(uniq, res)}
    return stem, {"native": native, "fold": fmap}


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from PACoAptamer.data_utils import load_s1
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = list(sp["test"])
    data = load_s1(set(test), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    stems = sorted(set(test) & set(data))
    stems = [s for s in stems if len(data[s]["seq"]) <= 512]
    print(f"[104e] 链数 {len(stems)}", flush=True)

    ref_pairs = {}
    for s in stems:
        rp = pairs_from_matrix(np.asarray(data[s]["ss_pairs"]))
        ref_pairs[s] = sorted([list(p) for p in rp])

    args = [(s, data[s]["seq"]) for s in stems]
    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, rec) in enumerate(ex.map(work_one, args, chunksize=4)):
            if rec is not None:
                per_chain[stem] = rec
            if (i + 1) % 50 == 0:
                print(f"[104e] {i+1}/{len(args)} | {time.time()-t0:.0f}s", flush=True)

    out = {"note": "gRNAde _cfix 池 RNAfold --noPS 回折缓存（2026-10-16）；"
                   "fold: {seq: [dotbracket, mfe]}；native/ref_pairs 为 s1 缓存口径",
           "ref_pairs": ref_pairs, "per_chain": per_chain}
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, OUT)
    n_seq = sum(len(v["fold"]) for v in per_chain.values())
    print(f"[104e] 完成 {len(per_chain)} 链，{n_seq} 唯一序列 | {time.time()-t0:.0f}s → {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
