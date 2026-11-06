#!/usr/bin/env python3
"""104b：69 号 A 层主表 gRNAde 行用 _cfix 池重算 → survey/69_alayer_gfix.json

口径与 data/69_alayer_retrained_tables.py 一致：
  native = grnade_out_v2/{stem}.native（manifest 口径）；池 = 3 温度拼接，t01 = 前 100 均值；
  iface_best/overall_best 用 69 同款 interface_positions（rcsb_pdb + eval_inputs_v2）。
仅把 grnade_out_av2_retrained 换成 grnade_out_av2_retrained_cfix。
其余方法行/视图从 survey/alayer_v2_retrained.json 原样拷贝。
"""
import os, sys, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
EI = os.path.join(SUR, "eval_inputs_v2")

import importlib.util
spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}
TEMPS = ("0.1", "0.5", "1.0")


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def interface_positions(pdb_id, chain, native):
    """69 号同款逻辑（逐字搬移）"""
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
        return None
    e = json.load(open(os.path.join(EI, f"{pdb_id}.json")))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    return set(i for i, (rn, rn_num, ic) in enumerate(seq_keys) if (rn_num, ic) in iface_keys)


def recs_of(seqs, native, pos_iface, t01_n=100):
    """69 号 recs_of 的 gRNAde 分支（skip_dash=False, nampnn=False）"""
    seqs = [s for s in seqs if len(s) == len(native)]
    if not seqs:
        return None
    recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in seqs]
    t01 = float(np.mean(recs[:t01_n])) if t01_n and len(recs) >= t01_n else float(np.mean(recs))
    ib = ov = None
    if pos_iface:
        ipos = sorted(pos_iface)
        ib = max(sum(1 for i in ipos if s[i] == native[i]) / len(ipos) for s in seqs)
        ov = max(recs)
    return (max(recs), t01, len(recs), ib, ov)


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    indep = sorted(sp["independent_test"])
    zs = sorted(sp["zero_shot_targets"])
    union = sorted(set(indep) | set(zs))

    pc_gr = {}
    for stem in union:
        native_p = os.path.join(SUR, "grnade_out_v2", f"{stem}.native")
        if not os.path.exists(native_p):
            continue
        native = open(native_p).read().strip()
        pool = []
        for t in TEMPS:
            fp = os.path.join(SUR, "grnade_out_av2_retrained_cfix", f"{stem}.{t}.seqs")
            if os.path.exists(fp):
                pool += [l.strip() for l in open(fp) if l.strip()]
        pdb, ch = stem.rsplit("_", 1)
        pos = interface_positions(pdb, ch, native)
        r = recs_of(pool, native, pos, t01_n=100)
        if r is None:
            continue
        pc_gr[stem] = {"recovery_best": r[0], "recovery_t01_mean": r[1], "n_samples": r[2],
                       "iface_best": r[3], "overall_best": r[4]}

    orig = json.load(open(os.path.join(SUR, "alayer_v2_retrained.json")))
    per_chain = {s: dict(v) for s, v in orig["per_chain"].items()}
    for stem, v in pc_gr.items():
        per_chain.setdefault(stem, {})["gRNAde"] = v

    def row_for(subset):
        vals = [per_chain[c]["gRNAde"] for c in subset if c in per_chain and "gRNAde" in per_chain[c]]
        best = float(np.mean([v["recovery_best"] for v in vals]))
        t01 = float(np.mean([v["recovery_t01_mean"] for v in vals]))
        iv = [v for v in vals if v["iface_best"] is not None]
        ir = float(np.mean([v["iface_best"] for v in iv])) if iv else None
        io = float(np.mean([v["overall_best"] for v in iv])) if iv else None
        return {"best": best, "t01": t01, "iface": ir, "iface_overall": io,
                "n": len(vals), "n_iface": len(iv)}

    views = {k: dict(v) for k, v in orig["views"].items()}
    views["independent_test"]["gRNAde"] = row_for(indep)
    views["zero_shot_targets"]["gRNAde"] = row_for(zs)
    views["union"]["gRNAde"] = row_for(union)

    out = {"views": views, "per_chain": per_chain,
           "cfix_note": ("gRNAde 行 = grnade_out_av2_retrained_cfix（17v2/26v2 解码 C/G 互换"
                         "修正后）按 69 号口径重算（2026-10-16）；其余方法拷贝自 "
                         "alayer_v2_retrained.json。修正前 gRNAde：indep best 0.491690/"
                         "t01 0.302133；zs best 0.506452/t01 0.311622。")}
    atomic_json(out, os.path.join(SUR, "69_alayer_gfix.json"))
    for view in ("independent_test", "zero_shot_targets", "union"):
        o = orig["views"][view]["gRNAde"]
        n = views[view]["gRNAde"]
        print(f"[104b] {view}: best {o['best']:.4f}->{n['best']:.4f}  "
              f"t01 {o['t01']:.4f}->{n['t01']:.4f}  "
              f"iface {o['iface'] if o['iface'] is None else round(o['iface'],4)}->"
              f"{n['iface'] if n['iface'] is None else round(n['iface'],4)} (n={n['n']})")


if __name__ == "__main__":
    main()
