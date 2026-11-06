#!/usr/bin/env python3
"""RNet 泄露审计 v2:RMDB 全库实测构建序列 vs 三测试层原生序列

背景:RibonanzaNet(RNet)训练于 Eterna/RMDB 化学映射数据(~200万条),
其中含 PDB/Rfam/Pseudobase 等天然来源子库;本项目 test/indep/zs 原生序列
全部来自 PDB。若原生序列(或其高相似/嵌入形式)出现在 RNet 训练数据中,
则 RNet 筛选实验存在记忆泄露,不得加入论文。

RMDB RDAT 两种实测序列载体,均提取:
  A. BigLib 型:ANNOTATION_DATA:N\tsequence:<ACUG>(含全部突变体, 已测序)
  B. 经典 M2 型:SEQUENCE 为野生型 + ANNOTATION_DATA mutation:G126C[,..]
     (按 OFFSET 换算位置, 重建单/双点突变体)

匹配口径(三级):
  A. 精确命中(U/T 归一后完全一致)
  B. 高相似(全局编辑距离相似度 >= 0.8)
  C. 嵌入包含(双向半局部对齐 HW, 编辑 <= 10% 长度)
预筛:12-mer 精确共享(numpy 2-bit 编码 + searchsorted, 随机序列期望共享
≈0.003 个 12-mer, >=2 个即几乎必然同源) → edlib 精验证。

输出:survey/89_rnet_leakage_audit.json
运行:纯 CPU,约几分钟。
"""
import os, sys, json, glob, time
from collections import defaultdict, Counter

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
RDAT_DIR = os.path.join(ROOT, "data", "rmdb_audit")

import numpy as np
import edlib

K = 12
CH2I = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


def norm_seq(s):
    s = "".join(s.split()).upper().replace("T", "U")
    return "".join(c for c in s if c in "AUCGN")


def kmers_u64(seq):
    """序列 → uint64 12-mer 数组(2-bit 编码)。非法字符(AUCG 以外)处
    切断,只从连续有效片段生成,绝不跨无效位拼接。"""
    out = []
    lut = np.full(256, 255, np.uint8)
    for c, v in CH2I.items():
        lut[ord(c)] = v
    # 切成 AUCG 连续片段
    frag = []
    cur = []
    for c in seq:
        if c in CH2I:
            cur.append(c)
        elif cur:
            frag.append("".join(cur)); cur = []
    if cur:
        frag.append("".join(cur))
    for fr in frag:
        v = lut[np.frombuffer(fr.encode(), dtype=np.uint8)]
        n = len(v)
        if n < K:
            continue
        acc = np.zeros(n - K + 1, np.uint64)
        for j in range(K):
            acc = (acc << np.uint64(2)) | v[j:j + n - K + 1].astype(np.uint64)
        out.append(acc)
    if not out:
        return np.empty(0, np.uint64)
    return np.unique(np.concatenate(out))


def parse_rdat(path):
    """返回 (sequences, n_mut_reconstructed)。"""
    seqs, muts = [], 0
    wt = None
    offset = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = line.rstrip("\n")
            if t.startswith("OFFSET\t"):
                try:
                    offset = int(t.split("\t")[1].split()[0])
                except Exception:
                    offset = 0
            elif t.startswith("SEQUENCE\t"):
                rest = t.split("\t", 1)[1] if "\t" in t else ""
                if set(rest) & set("AUCG"):
                    wt = norm_seq(rest)
            elif t.startswith("ANNOTATION_DATA:"):
                for field in t.split("\t")[1:]:
                    if field.startswith("sequence:"):
                        s = norm_seq(field[len("sequence:"):])
                        if len(s) >= 15:
                            seqs.append(s)
                    elif field.startswith("mutation:") and wt is not None:
                        mut = field[len("mutation:"):]
                        if mut == "WT":
                            continue
                        seq = list(wt)
                        ok = True
                        for m in mut.split(","):
                            m = m.strip()
                            # 格式: 参考碱基+位置+新碱基, 如 G126C
                            if (len(m) >= 4 and m[0] in "ACGU"
                                    and m[1:-1].isdigit() and m[-1] in "ACGU"):
                                p = int(m[1:-1]) - offset - 1
                                if 0 <= p < len(seq):
                                    seq[p] = m[-1]
                                else:
                                    ok = False
                            else:
                                ok = False
                        if ok:
                            seqs.append("".join(seq))
                            muts += 1
    if wt and len(wt) >= 15:
        seqs.append(wt)
    return seqs, muts


def main():
    t0 = time.time()
    # ---- 1) 原生序列与 12-mer 索引 ----
    # 原生序列取自 splits_v2 自带的规范序列汇总(canonical truth,
    # 完整覆盖三层; paco_s1_v2 缓存缺 indep/zs 的 2024+ 链)
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    seqs_all = json.load(open(os.path.join(ROOT, "data", "splits_v2",
                                           "seqs_all.json")))
    layers = {"test": "test", "indep": "independent_test", "zs": "zero_shot_targets"}
    natives = []          # [(layer, stem, seq)]
    seen = set()
    for layer, key in layers.items():
        for stem in sorted(set(sp[key]) & set(seqs_all)):
            s = norm_seq(seqs_all[stem])
            if len(s) >= 15 and (layer, s) not in seen:
                seen.add((layer, s))
                natives.append((layer, stem, s))
    print(f"[audit] natives={len(natives)} "
          f"(layer sizes: " + ", ".join(
              f"{l}={sum(1 for x in natives if x[0]==l)}"
              for l in layers) + ")", flush=True)

    kmer2nat = defaultdict(list)
    for i, (_, _, s) in enumerate(natives):
        for km in np.unique(kmers_u64(s)):
            kmer2nat[int(km)].append(i)
    nat_kmers_sorted = np.array(sorted(kmer2nat), np.uint64)
    print(f"[audit] native 12-mers={len(nat_kmers_sorted)}", flush=True)
    exact_set = {s: i for i, (_, _, s) in enumerate(natives)}

    # ---- 2) 流式扫描 RMDB ----
    files = sorted(glob.glob(os.path.join(RDAT_DIR, "*.rdat")))
    n_expect = sum(len(open(os.path.join(RDAT_DIR, f)).readlines())
                   for f in glob.glob(os.path.join(RDAT_DIR, "urls_*.txt")))
    partial = os.environ.get("ALLOW_PARTIAL") != "1" and len(files) < n_expect
    if partial:
        print(f"[audit] ⚠️ RDAT 不全({len(files)}/{n_expect}),本次仅试运行,"
              "不写正式结论", flush=True)
    cand = Counter()                      # (construct_idx, native_idx) -> 共享数
    cons_seqs = []                        # 候选构建序列(全局编号)
    cons_src = []                         # 来源 RDAT
    n_seqs = n_files_bad = n_mut = 0
    for fi, path in enumerate(files):
        rid = os.path.basename(path)[:-5]
        try:
            seqs, muts = parse_rdat(path)
        except Exception:
            n_files_bad += 1
            continue
        n_mut += muts
        n_seqs += len(seqs)
        for s in seqs:
            if s in exact_set:
                j = len(cons_seqs)
                cons_seqs.append(s); cons_src.append(rid)
                cand[(j, exact_set[s])] = 10 ** 6   # 精确命中直推
                continue
            km = kmers_u64(s)
            if not len(km):
                continue
            pos = np.searchsorted(nat_kmers_sorted, km)
            pos_c = np.minimum(pos, len(nat_kmers_sorted) - 1)
            match = nat_kmers_sorted[pos_c] == km
            if not match.any():
                continue
            nat_ids = Counter()
            for m in km[match]:
                for ni in kmer2nat[int(m)]:
                    nat_ids[ni] += 1
            good = [(ni, c) for ni, c in nat_ids.items() if c >= 2]
            if not good:
                continue                     # 随机单碰撞,不存序列
            li = len(cons_seqs)
            cons_seqs.append(s); cons_src.append(rid)
            for ni, c in good:
                cand[(li, ni)] += c
        if (fi + 1) % 100 == 0:
            print(f"[audit] {fi+1}/{len(files)} files | {n_seqs} seqs | "
                  f"pairs={len(cand)} | {time.time()-t0:.0f}s", flush=True)

    print(f"[audit] scan done: files={len(files)} bad={n_files_bad} "
          f"seqs={n_seqs} (mut_reconstructed={n_mut}) "
          f"candidate_pairs={len(cand)} | {time.time()-t0:.0f}s", flush=True)

    # ---- 3) edlib 精验证 ----
    exact, high, embedded, frag = {}, {}, {}, {}
    for (ci, ni), shared in cand.items():
        cs, nat = cons_seqs[ci], natives[ni][2]
        key = f"{natives[ni][0]}:{natives[ni][1]}"
        src = cons_src[ci]
        if shared >= 10 ** 6:
            exact.setdefault(key, []).append({"rmdb": src, "len": len(cs)})
            continue
        d = edlib.align(nat, cs, task="distance")["editDistance"]
        sim = 1 - d / max(len(nat), len(cs))
        if sim >= 0.8:
            high.setdefault(key, []).append(
                {"rmdb": src, "sim": round(sim, 3), "shared12": shared})
            continue
        dh = edlib.align(nat, cs, mode="HW", task="distance")["editDistance"]
        dr = edlib.align(cs, nat, mode="HW", task="distance")["editDistance"]
        if dh / len(nat) <= 0.1:
            embedded.setdefault(key, []).append(
                {"rmdb": src, "edit": dh, "len": len(cs), "shared12": shared})
        elif dr / len(cs) <= 0.1:
            embedded.setdefault(key, []).append(
                {"rmdb": src, "reverse_edit": dr, "len": len(cs),
                 "shared12": shared})
        elif shared >= 6:
            frag.setdefault(key, []).append(
                {"rmdb": src, "shared12": shared, "len": len(cs),
                 "sim": round(sim, 3)})

    if partial:
        print(f"[audit] 试运行结束(文件不全,未写 JSON)。"
              f"exact={len(exact)} high={len(high)} embedded={len(embedded)} "
              f"frag={len(frag)}")
        return
    out = {
        "rdat_files": len(files), "rdat_bad": n_files_bad,
        "n_sequences_scanned": n_seqs, "n_mutations_reconstructed": n_mut,
        "n_natives": len(natives),
        "n_exact": len(exact), "exact": exact,
        "n_high": len(high), "high": high,
        "n_embedded": len(embedded), "embedded": embedded,
        "n_fragment_suspect": len(frag), "fragment_suspect": frag,
        "verdict": "PASS" if not (exact or high or embedded) else "FAIL",
        "criteria": {
            "exact": "identical after U/T normalization",
            "high": "global edit-distance similarity >= 0.8",
            "embedded": "either-direction HW alignment, <=10% edit",
            "fragment_suspect": ">=6 shared 12-mers (reported, not leakage "
                                "per se)",
            "prefilter": ">=2 shared 12-mers (2-bit encoded)"},
    }
    json.dump(out, open(os.path.join(SUR, "89_rnet_leakage_audit.json"), "w"),
              indent=1)
    print(f"[audit] exact={len(exact)} high={len(high)} embedded={len(embedded)} "
          f"frag={len(frag)} verdict={out['verdict']} | "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
