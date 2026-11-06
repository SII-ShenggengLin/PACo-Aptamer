#!/usr/bin/env python3
"""104i：77_p02 距离梯度的 gRNAde 行用 _cfix 池重算 → survey/77_p02_gfix.json

口径与 data/77_p0234_analysis.py phase_dist 完全一致：
  115 界面链（74_iface_meta usable 且 0<n_iface<=50）；逐残基距配体最近原子距离
  （m35.extract_rna → difflib 映射到 s1 native 坐标系）；五距离分箱；
  池 = _cfix 三温度拼接（len==native 过滤，<5 序列跳过）；
  每箱 mean/median/boot_ci(2000,seed=0) + 归一化(bin−链均值) + iface−>20Å 配对差。
自检：同管线重算 PACo_ens2 行并与原 JSON 逐键比对（应完全一致），仅作验证不写入。
其余方法行从 77_p02_distance_gradient.json 原样拷贝。
环境：mpl372_repro + PYTHONPATH=data/survey/pyshims
"""
import os, sys, json, difflib
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m35 = _imp("m35", os.path.join(DATA_DIR, "35_build_s2_samples.py"))

BINS = [(0.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 1e9)]
BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]
TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100
POOL_DIR = os.path.join(SUR, "77_ens2_pool_btest")
OUT = os.path.join(SUR, "77_p02_gfix.json")


def boot_ci(x, n=2000, seed=0):
    if len(x) < 3:
        return None
    rng = np.random.RandomState(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def compute_dist_native():
    """phase_dist 的 dist_native 构建逐字复刻"""
    from scipy.spatial import cKDTree
    from PACoAptamer.data_utils import load_s1
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    stems = list(sp2["test"])
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    usable = {k: v for k, v in meta.items()
              if v and v.get("usable") and 0 < v["n_iface"] <= 50}
    dist_native = {}
    for stem, mv in usable.items():
        pdb_id, ch = stem.rsplit("_", 1)
        native = data[stem]['seq'] if stem in data else None
        if native is None:
            continue
        residues, seq35, _ = m35.extract_rna(pdb_id, ch)
        if seq35 is None or len(residues) != len(seq35):
            continue
        if seq35 == native:
            mapping = list(range(len(native)))
        else:
            sm = difflib.SequenceMatcher(None, seq35, native, autojunk=False)
            m2n = {}
            ok = True
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    m2n[a + k] = b + k
            for i in range(len(seq35)):
                if i not in m2n:
                    ok = False
                    break
            if not ok or len(m2n) != len(native):
                continue
            mapping = [m2n[i] for i in range(len(seq35))]
        tree_lig = cKDTree(np.array(mv["lig_coords"], dtype=np.float32))
        dists = np.full(len(native), np.nan)
        for i, (_, _, _, atoms) in enumerate(residues):
            pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
            if len(pts):
                dd, _ = tree_lig.query(pts, k=1)
                dists[mapping[i]] = float(dd.min())
        dist_native[stem] = {"native": native, "dist": dists}
    return dist_native


def method_rows(dist_native, pool_fn):
    """phase_dist 每方法聚合逐字复刻"""
    per_chain = {}
    for stem, dv in dist_native.items():
        native, dists = dv["native"], dv["dist"]
        valid = np.array([c != "N" and np.isfinite(dists[i])
                          for i, c in enumerate(native)])
        try:
            pool = pool_fn(stem)
        except FileNotFoundError:
            pool = None
        if not pool:
            continue
        seqs = [s for s in pool if len(s) == len(native)]
        if len(seqs) < 5:
            continue
        mat = np.array([list(s) for s in seqs])
        rec = (mat == np.array(list(native))).mean(axis=0)
        rec_by_bin, n_by_bin = {}, {}
        for (lo, hi), nm in zip(BINS, BIN_NAMES):
            sel = valid & (dists >= lo) & (dists < hi)
            if sel.sum() > 0:
                rec_by_bin[nm] = float(rec[sel].mean())
                n_by_bin[nm] = int(sel.sum())
        glob = float(rec[valid].mean())
        per_chain[stem] = {"rec": rec_by_bin, "n": n_by_bin,
                           "n_seq": len(seqs), "glob": glob}
    rows = {}
    for nm in BIN_NAMES:
        vals = [c["rec"][nm] for c in per_chain.values() if nm in c["rec"]]
        if vals:
            rows[nm] = {"n_chains": len(vals), "mean": float(np.mean(vals)),
                        "ci95": boot_ci(vals), "median": float(np.median(vals))}
        nvals = [c["rec"][nm] - c["glob"] for c in per_chain.values() if nm in c["rec"]]
        if nvals:
            rows[nm]["norm_mean"] = float(np.mean(nvals))
            rows[nm]["norm_ci95"] = boot_ci(nvals)
    difs = [c["rec"]["≤5Å(界面)"] - c["rec"][">20Å"] for c in per_chain.values()
            if "≤5Å(界面)" in c["rec"] and ">20Å" in c["rec"]]
    rows["iface_minus_gt20"] = {"n": len(difs), "mean": float(np.mean(difs)),
                                "ci95": boot_ci(difs)} if difs else None
    rows["n_chains_total"] = len(per_chain)
    return rows


def main():
    dist_native = compute_dist_native()
    print(f"[104i] 距离可算链 {len(dist_native)}", flush=True)

    gr_rows = method_rows(dist_native, lambda s: [
        l.strip() for t in TEMPS
        for l in open(os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{s}.{t}.seqs"))])

    # 自检：PACo_ens2 行应与原 JSON 完全一致
    def paco_pool(s):
        p = os.path.join(POOL_DIR, s + ".json")
        return json.load(open(p))["pool"] if os.path.exists(p) else None
    paco_rows = method_rows(dist_native, paco_pool)

    orig = json.load(open(os.path.join(SUR, "77_p02_distance_gradient.json")))
    o_paco = orig["methods"]["PACo_ens2"]
    ok = True
    for nm in BIN_NAMES:
        a, b = paco_rows[nm]["mean"], o_paco[nm]["mean"]
        if abs(a - b) > 1e-9:
            ok = False
            print(f"  [自检 FAIL] PACo {nm}: {a} vs {b}")
    if abs(paco_rows["iface_minus_gt20"]["mean"] - o_paco["iface_minus_gt20"]["mean"]) > 1e-9:
        ok = False
        print("  [自检 FAIL] PACo iface_minus_gt20")
    print(f"[104i] PACo 自检 {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok

    out = {"n_iface_chains_dist": orig["n_iface_chains_dist"],
           "methods": dict(orig["methods"])}
    out["methods"]["gRNAde"] = gr_rows
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix 池按 77 号 phase_dist 口径重算"
                        "（2026-10-16）；其余行拷贝自 77_p02_distance_gradient.json。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    o = orig["methods"]["gRNAde"]
    for nm in BIN_NAMES:
        print(f"[104i] gRNAde {nm}: {o[nm]['mean']:.4f}->{gr_rows[nm]['mean']:.4f}")
    print(f"       界面−远端 Δ {o['iface_minus_gt20']['mean']:+.4f}->"
          f"{gr_rows['iface_minus_gt20']['mean']:+.4f} (n={gr_rows['n_chains_total']})")


if __name__ == "__main__":
    main()
