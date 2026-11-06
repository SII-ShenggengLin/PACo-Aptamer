#!/usr/bin/env python3
"""104l：92 号界面 logo 的 gRNAde 频率矩阵用 _cfix t=0.1 池重算
       → survey/92_interface_logos_gfix.json

口径与 data/92_interface_logos.py 完全一致（freq_matrix / summarize_methods / 115 链汇总
逐字复刻），仅 gRNAde 的池换成 grnade_out_btest_v2_cfix/{stem}.0.1.seqs。
其余方法的 case_detail / case_summary / all115_summary 从 92_interface_logos.json 原样拷贝。
环境：rna_if（纯 numpy）
"""
import os, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
BASES = ["A", "U", "G", "C"]
OUT = os.path.join(SUR, "92_interface_logos_gfix.json")


def load_pool_cfix(stem):
    p = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.0.1.seqs")
    if not os.path.exists(p):
        return []
    return [l.strip() for l in open(p) if l.strip()]


def freq_matrix(pool, native, iface_pos):
    L = len(native)
    valid = [s for s in pool if len(s) == L]
    per_pos = {}
    for i in iface_pos:
        cnt = {b: 0 for b in BASES}
        n_other = 0
        n_match = 0
        for s in valid:
            c = s[i].upper()
            if c in cnt:
                cnt[c] += 1
            else:
                n_other += 1
            if c == native[i]:
                n_match += 1
        tot = sum(cnt.values())
        freqs = {b: (cnt[b] / tot if tot else None) for b in BASES}
        consensus = max(BASES, key=lambda b: cnt[b]) if tot else None
        per_pos[str(i)] = {
            "native": native[i],
            "freq": freqs,
            "counts": cnt,
            "n_seqs": len(valid),
            "n_other_letter": n_other,
            "match_frac": (n_match / len(valid)) if valid else None,
            "consensus": consensus,
            "consensus_eq_native": (consensus == native[i]) if consensus else None,
        }
    return per_pos, len(valid)


def main():
    orig = json.load(open(os.path.join(SUR, "92_interface_logos.json")))
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    usable = {k: v for k, v in meta.items()
              if v and v.get("usable") and 0 < v["n_iface"] <= 50}
    assert len(usable) == 115

    # ---- case_detail：12 案例链的 gRNAde per_pos 重算
    detail = {s: dict(v) for s, v in orig["case_detail"].items()}
    for stem, rec in detail.items():
        native = rec["native_seq"]
        iface_pos = rec["iface_positions"]
        pool = load_pool_cfix(stem)
        if not pool:
            continue
        per_pos, n_valid = freq_matrix(pool, native, iface_pos)
        rec["per_method"] = dict(rec["per_method"])
        rec["per_method"]["gRNAde"] = {"n_pool": len(pool), "n_valid_len": n_valid,
                                       "per_pos": per_pos}

    # ---- case_summary：gRNAde 行按 summarize_methods 同款逻辑重算
    def summ_row(chain_data, m):
        micro_hits = micro_tot = 0
        macro, match_means = [], []
        for stem, rec in chain_data.items():
            pm = rec["per_method"].get(m)
            if pm is None:
                continue
            hits = tot = 0
            mf = []
            for pos, d in pm["per_pos"].items():
                if d["consensus_eq_native"] is None:
                    continue
                tot += 1
                hits += int(d["consensus_eq_native"])
                if d["match_frac"] is not None:
                    mf.append(d["match_frac"])
            micro_hits += hits
            micro_tot += tot
            if tot:
                macro.append(hits / tot)
            if mf:
                match_means.append(float(np.mean(mf)))
        return {
            "n_chains": len(macro),
            "consensus_eq_native_micro": (micro_hits / micro_tot) if micro_tot else None,
            "consensus_eq_native_macro_mean": float(np.mean(macro)) if macro else None,
            "match_frac_mean": float(np.mean(match_means)) if match_means else None,
            "n_positions": micro_tot,
        }

    case_summary = dict(orig["case_summary"])
    case_summary["gRNAde"] = summ_row(detail, "gRNAde")

    # ---- all115：gRNAde 行重算
    micro_hits = micro_tot = 0
    macro, match_means = [], []
    n_chains = 0
    for stem, v in sorted(usable.items()):
        pool = load_pool_cfix(stem)
        if not pool:
            continue
        pj = os.path.join(SUR, "77_ens2_pool_btest", f"{stem}.json")
        if not os.path.exists(pj):
            continue
        native = json.load(open(pj))["native"]
        if len(native) != len(v["interface"]):
            continue
        iface_pos = [i for i, x in enumerate(v["interface"]) if x]
        per_pos, _ = freq_matrix(pool, native, iface_pos)
        hits = tot = 0
        mf = []
        for d in per_pos.values():
            if d["consensus_eq_native"] is None:
                continue
            tot += 1
            hits += int(d["consensus_eq_native"])
            if d["match_frac"] is not None:
                mf.append(d["match_frac"])
        micro_hits += hits
        micro_tot += tot
        if tot:
            macro.append(hits / tot)
            n_chains += 1
        if mf:
            match_means.append(float(np.mean(mf)))
    all115 = dict(orig["all115_summary"])
    all115["gRNAde"] = {
        "n_chains": n_chains,
        "n_positions": micro_tot,
        "consensus_eq_native_micro": (micro_hits / micro_tot) if micro_tot else None,
        "consensus_eq_native_macro_mean": float(np.mean(macro)) if macro else None,
        "match_frac_mean": float(np.mean(match_means)) if match_means else None,
    }

    out = {"definition": orig["definition"], "case_detail": detail,
           "case_summary": case_summary, "all115_summary": all115,
           "cfix_note": ("gRNAde 频率矩阵/汇总 = grnade_out_btest_v2_cfix t=0.1 池按 92 号口径"
                         "重算（2026-10-16）；其余方法拷贝自 92_interface_logos.json。")}
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    o, n = orig["all115_summary"]["gRNAde"], all115["gRNAde"]
    print(f"[104l] 115链 gRNAde: 共识=native micro {o['consensus_eq_native_micro']:.4f}->"
          f"{n['consensus_eq_native_micro']:.4f} macro {o['consensus_eq_native_macro_mean']:.4f}->"
          f"{n['consensus_eq_native_macro_mean']:.4f} match {o['match_frac_mean']:.4f}->"
          f"{n['match_frac_mean']:.4f}")
    o, n = orig["case_summary"]["gRNAde"], case_summary["gRNAde"]
    print(f"       12案例 gRNAde: micro {o['consensus_eq_native_micro']:.4f}->"
          f"{n['consensus_eq_native_micro']:.4f} macro {o['consensus_eq_native_macro_mean']:.4f}->"
          f"{n['consensus_eq_native_macro_mean']:.4f}")


if __name__ == "__main__":
    main()
