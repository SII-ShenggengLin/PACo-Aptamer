#!/usr/bin/env python3
"""第 92 步：界面位点序列 logo 频率矩阵（实验 4，纯 CPU）

口径（与 74 号一致）：界面链 = survey/74_iface_meta.json 中 usable 且 0<n_iface<=50
的 115 条 test 链；界面位点 = interface 布尔数组中为 True 的 native 序列索引
（功能配体 HETATM + 残基原子 5Å 接触，74 号 meta 相位同款定义）。

案例链：论文 Fig.6a 画廊 12 条（figures/fig6/fig6_gallery.py 的 CASES）。其中
3FU2_C 在 74 口径下 unusable、6PRV_D 无功能配体条目 → 用界面残基数最多的非画廊链
补足 12 条（provenance 字段标注 gallery / backfill）。

每案例链 × 每方法：
  PACo      77_ens2_pool_btest/{stem}.json 的 pool 前 100 条（t=0.1 子池）
  RhoDesign rhodesign_out_btest_v2/{stem}.0.1.seqs
  gRNAde    grnade_out_btest_v2/{stem}.0.1.seqs
  NA-MPNN   nampnn_out_btest_v2/{stem}.seqs（全池）
  AlignIF   alignif_out_btest_v2/{stem}/out.fasta（全池）
计算每个界面位点的 A/U/G/C 频率矩阵（对 A/U/G/C 计数归一）+ native 碱基 + 匹配比例
+ 共识碱基。另在全部 115 条界面链上汇总"共识碱基=native"比例（每方法）。

输出：survey/92_interface_logos.json（临时文件 + rename）
环境：任意带 numpy 的 python（/root/miniconda3/envs/mpl372_repro/bin/python）
"""
import os, json, glob
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
OUT_TMP = os.path.join(SUR, "92_interface_logos.json.tmp")
OUT = os.path.join(SUR, "92_interface_logos.json")

GALLERY12 = ["3GAO_A", "6UC7_B", "7Q80_E", "3FU2_C", "8D5L_A", "6UET_A",
             "5BTP_B", "8HB1_A", "4K31_B", "4FE5_B", "6PRV_D", "6E8S_B"]
METHODS = ["PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
BASES = ["A", "U", "G", "C"]
N_T01 = 100   # PACo 池前 100 条 = t=0.1 子池（n_per_temp=100）


def load_pool(stem, method):
    """返回该方法的设计序列列表（只保留非空行；AlignIF 去 fasta 头）"""
    if method == "PACo":
        p = os.path.join(SUR, "77_ens2_pool_btest", f"{stem}.json")
        if not os.path.exists(p):
            return []
        return json.load(open(p))["pool"][:N_T01]
    if method in ("RhoDesign", "gRNAde"):
        d = "rhodesign_out_btest_v2" if method == "RhoDesign" else "grnade_out_btest_v2"
        p = os.path.join(SUR, d, f"{stem}.0.1.seqs")
        if not os.path.exists(p):
            return []
        return [l.strip() for l in open(p) if l.strip()]
    if method == "NA-MPNN":
        p = os.path.join(SUR, "nampnn_out_btest_v2", f"{stem}.seqs")
        if not os.path.exists(p):
            return []
        return [l.strip() for l in open(p) if l.strip()]
    if method == "AlignIF":
        p = os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta")
        if not os.path.exists(p):
            return []
        return [l.strip() for l in open(p) if l.strip() and not l.startswith(">")]
    raise ValueError(method)


def freq_matrix(pool, native, iface_pos):
    """每位 A/U/G/C 频率（对有效字母计数归一）+ native 匹配比例 + 共识"""
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


def summarize_methods(chain_data):
    """对 case_detail：每方法 micro/macro 共识=native 比例 + 平均匹配率"""
    summ = {}
    for m in METHODS:
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
        summ[m] = {
            "n_chains": len(macro),
            "consensus_eq_native_micro": (micro_hits / micro_tot) if micro_tot else None,
            "consensus_eq_native_macro_mean": float(np.mean(macro)) if macro else None,
            "match_frac_mean": float(np.mean(match_means)) if match_means else None,
            "n_positions": micro_tot,
        }
    return summ


def main():
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    usable = {k: v for k, v in meta.items()
              if v and v.get("usable") and 0 < v["n_iface"] <= 50}
    assert len(usable) == 115, f"界面链数 {len(usable)} != 115"

    # ---- 案例链：画廊 12 条，缺 2 条按 n_iface 补足
    cases, provenance = [], {}
    for s in GALLERY12:
        if s in usable:
            cases.append(s)
            provenance[s] = "gallery"
        else:
            v = meta.get(s)
            why = "no_entry" if v is None else \
                  ("unusable" if not v.get("usable") else f"n_iface={v.get('n_iface')}")
            print(f"画廊链 {s} 不在 115 界面链内（{why}），跳过")
    backfill = sorted((s for s in usable if s not in cases),
                      key=lambda s: -usable[s]["n_iface"])[:12 - len(cases)]
    for s in backfill:
        cases.append(s)
        provenance[s] = f"backfill(n_iface={usable[s]['n_iface']})"
    print(f"案例链 {len(cases)} 条: {cases}")

    # ---- 案例链明细
    detail = {}
    for stem in cases:
        v = usable[stem]
        iface_pos = [i for i, x in enumerate(v["interface"]) if x]
        native = None
        per_method = {}
        for m in METHODS:
            pool = load_pool(stem, m)
            if m == "PACo":
                native = json.load(open(os.path.join(
                    SUR, "77_ens2_pool_btest", f"{stem}.json")))["native"]
            if not pool:
                print(f"  [warn] {stem}/{m}: 池缺失")
                continue
            if native is None:
                native = pool and None
        # native 以 PACo 池为准；缺失时用 74 meta 无法恢复序列 → 报错
        assert native is not None and len(native) == len(v["interface"]), stem
        for m in METHODS:
            pool = load_pool(stem, m)
            if not pool:
                continue
            per_pos, n_valid = freq_matrix(pool, native, iface_pos)
            per_method[m] = {"n_pool": len(pool), "n_valid_len": n_valid,
                             "per_pos": per_pos}
        detail[stem] = {
            "provenance": provenance[stem],
            "native_seq": native,
            "iface_positions": iface_pos,
            "n_iface": len(iface_pos),
            "per_method": per_method,
        }
        print(f"  {stem}: L={len(native)} n_iface={len(iface_pos)} "
              f"pool方法={list(per_method)}")

    case_summary = summarize_methods(detail)

    # ---- 全部 115 链：共识=native 汇总（每方法）
    all115 = {}
    for m in METHODS:
        micro_hits = micro_tot = 0
        macro = []
        match_means = []
        n_chains = 0
        for stem, v in sorted(usable.items()):
            pool = load_pool(stem, m)
            if not pool:
                continue
            # native：74 meta 不存序列 → 从 PACo 池 json 读（与 74 同 s1 缓存口径）
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
        all115[m] = {
            "n_chains": n_chains,
            "n_positions": micro_tot,
            "consensus_eq_native_micro": (micro_hits / micro_tot) if micro_tot else None,
            "consensus_eq_native_macro_mean": float(np.mean(macro)) if macro else None,
            "match_frac_mean": float(np.mean(match_means)) if match_means else None,
        }
        print(f"  [115链] {m}: micro={all115[m]['consensus_eq_native_micro']:.4f} "
              f"macro={all115[m]['consensus_eq_native_macro_mean']:.4f} "
              f"match={all115[m]['match_frac_mean']:.4f} (n={n_chains})")

    out = {
        "definition": {
            "iface_source": "survey/74_iface_meta.json（74 号 meta 相位：功能配体 HETATM "
                            "+ 残基原子 5Å 接触；usable 且 0<n_iface<=50）",
            "n_iface_chains": len(usable),
            "gallery12": GALLERY12,
            "cases_used": cases,
            "pools": {"PACo": "77_ens2_pool_btest pool[:100]（t=0.1 子池）",
                      "RhoDesign/gRNAde": "{stem}.0.1.seqs",
                      "NA-MPNN/AlignIF": "全池"},
            "freq_norm": "A/U/G/C 计数归一；非 AUGC 字母计入 n_other_letter",
        },
        "case_detail": detail,
        "case_summary": case_summary,
        "all115_summary": all115,
    }
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, OUT)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
