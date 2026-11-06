#!/usr/bin/env python3
"""第 82 步：Fig5a 温度-多样性权衡 + Fig5e GC/MFE 序列可行性（蓝图 §五 P1-7）

Fig5a：三温度下 recovery vs 池内多样性（PACo ens2 池切片 + RhoDesign/gRNAde 分温度文件）
       多样性 = 池内平均两两序列一致度（1−identity）+ 唯一序列占比
Fig5e：t=0.1 池逐设计 GC 含量与 MFE/nt（RNAfold），五方法 + 天然参照
输出：survey/82_temp_diversity.json、survey/82_gc_mfe.json
"""
import os, sys, json, time
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m77 = _imp("m77", os.path.join(ROOT, "data", "77_p0234_analysis.py"))
TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100


def pool_diversity(seqs, k=50):
    """平均两两一致度 → 多样性 = 1 − identity；+ 唯一占比"""
    if len(seqs) < 2:
        return None, None
    uniq = len(set(seqs)) / len(seqs)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(seqs))[:min(k, len(seqs))]
    sub = [seqs[i] for i in idx]
    ids = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            a, b = sub[i], sub[j]
            if len(a) != len(b):
                continue
            ids.append(sum(1 for x, y in zip(a, b) if x == y) / len(a))
    return (float(1 - np.mean(ids)) if ids else None), float(uniq)


def phase_temp():
    stems = m77.test_stems()
    from PACoAptamer.data_utils import load_s1
    data = load_s1(set(stems), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))   # 只加载一次
    out = {}
    for m in ("PACo", "RhoDesign", "gRNAde"):
        per_temp = {t: {"rec": [], "div": [], "uniq": []} for t in TEMPS}
        for stem in stems:
            if m == "PACo":
                p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
                if not os.path.exists(p):
                    continue
                v = json.load(open(p))
                pools = {t: v["pool"][i * N_PER_TEMP:(i + 1) * N_PER_TEMP]
                         for i, t in enumerate(TEMPS)}
                native = v["native"]
            else:
                d = f"{SUR}/{'rhodesign' if m=='RhoDesign' else 'grnade'}_out_btest_v2"
                pools = {}
                if stem not in data:
                    continue
                native = data[stem]["seq"]
                for t in TEMPS:
                    fp = os.path.join(d, f"{stem}.{t}.seqs")
                    if not os.path.exists(fp):
                        continue
                    pools[t] = [l.strip() for l in open(fp)]
            for t, pool in pools.items():
                pool = [s for s in pool if len(s) == len(native)]
                if len(pool) < 10:
                    continue
                recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
                div, uniq = pool_diversity(pool)
                per_temp[t]["rec"].append(float(np.mean(recs)))
                if div is not None:
                    per_temp[t]["div"].append(div)
                per_temp[t]["uniq"].append(uniq)
        out[m] = {}
        for t, pt in per_temp.items():
            out[m][f"t={t}"] = {k: {"mean": float(np.mean(v)) if v else None,
                                    "n": len(v)} for k, v in pt.items()}
    json.dump(out, open(os.path.join(SUR, "82_temp_diversity.json"), "w"), indent=1)
    for m, temps in out.items():
        line = " | ".join(f"t={t}: rec {temps[f't={t}']['rec']['mean']:.3f} "
                          f"div {temps[f't={t}']['div']['mean']:.3f} "
                          f"uniq {temps[f't={t}']['uniq']['mean']:.3f}" for t in TEMPS)
        print(f"[82-temp] {m}: {line}")


def _gcmfe_worker(args):
    stem, native = args
    pools = {}
    p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
    if os.path.exists(p):
        v = json.load(open(p))
        pools["PACo"] = [s for s in v["pool"][:N_PER_TEMP] if len(s) == len(native)]
    for m, d in (("RhoDesign", "rhodesign_out_btest_v2"), ("gRNAde", "grnade_out_btest_v2")):
        fp = os.path.join(SUR, d, f"{stem}.0.1.seqs")
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp) if len(l.strip()) == len(native)]
    for m, fp in (("NA-MPNN", os.path.join(SUR, "nampnn_out_btest_v2", stem + ".seqs")),
                  ("AlignIF", os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta"))):
        if os.path.exists(fp):
            pools[m] = [l.strip() for l in open(fp)
                        if not l.startswith(">") and len(l.strip()) == len(native)]
    out = {}
    for m, pool in pools.items():
        if not pool:
            continue
        gc = [(s.count("G") + s.count("C")) / len(s) for s in pool]
        uniq = sorted(set(s for s in pool if set(s) <= set("ACGU")))
        mfes = mfe_of(uniq) if uniq else []
        mfes = [m / len(s) for m, s in zip(mfes, uniq)]   # → 每核苷酸
        out[m] = {"gc_mean": float(np.mean(gc)), "n": len(pool),
                  "mfe_nt_mean": float(np.mean(mfes)) if mfes else None}
    # native 参照（mfe_of 解析失败返回 [] 时不再崩 None/int，2026-08-19 修复）
    nat_mfe = (mfe_of([native]) or [None])[0]
    out["_native"] = {"gc": (native.count("G") + native.count("C")) / len(native),
                      "mfe_nt": (nat_mfe / len(native)) if nat_mfe is not None else None}
    return stem, out


def mfe_of(seqs):
    """RNAfold 批量，解析 MFE (kcal/mol) → 每核苷酸"""
    import subprocess, tempfile, re
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        f.write("\n".join(seqs) + "\n")
        fn = f.name
    try:
        r = subprocess.run([m77.RNAFOLD, "--noPS"], stdin=open(fn),
                           capture_output=True, text=True, timeout=600)
        out = []
        for l in r.stdout.splitlines():
            m = re.search(r"\(\s*(-?\d+\.\d+)\)\s*$", l)
            if m:
                out.append(float(m.group(1)))
        return out if len(out) == len(seqs) else []
    finally:
        os.unlink(fn)


def phase_gcmfe():
    from PACoAptamer.data_utils import load_s1
    from concurrent.futures import ProcessPoolExecutor
    stems = m77.test_stems()
    data = load_s1(set(stems), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    args = [(s, data[s]["seq"]) for s in sorted(set(stems) & set(data))
            if len(data[s]["seq"]) <= 512]
    per_chain = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (stem, rec) in enumerate(ex.map(_gcmfe_worker, args, chunksize=4)):
            per_chain[stem] = rec
            if (i + 1) % 100 == 0:
                print(f"[82-gcmfe] {i+1}/{len(args)} | {time.time()-t0:.0f}s", flush=True)
    out = {"methods": {}}
    for m in ("PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"):
        gcs = [v[m]["gc_mean"] for v in per_chain.values() if m in v]
        mfes = [v[m]["mfe_nt_mean"] for v in per_chain.values()
                if m in v and v[m].get("mfe_nt_mean") is not None]
        out["methods"][m] = {"gc_mean": float(np.mean(gcs)), "n": len(gcs),
                             "mfe_nt_mean": float(np.mean(mfes)) if mfes else None,
                             "mfe_nt_median": float(np.median(mfes)) if mfes else None}
    ngc = [v["_native"]["gc"] for v in per_chain.values() if "_native" in v]
    nmfe = [v["_native"]["mfe_nt"] for v in per_chain.values()
            if v.get("_native", {}).get("mfe_nt") is not None]
    out["native_ref"] = {"gc_mean": float(np.mean(ngc)), "n": len(ngc),
                         "mfe_nt_mean": float(np.mean(nmfe)) if nmfe else None}
    json.dump(out, open(os.path.join(SUR, "82_gc_mfe.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['temp', 'gcmfe'])
    a = ap.parse_args()
    dict(temp=phase_temp, gcmfe=phase_gcmfe)[a.phase]()


if __name__ == "__main__":
    main()
