#!/usr/bin/env python3
"""第 69 步：A 层 v2 重训权重指标汇总（indep / zero_shot / 合并 三视图 + 界面）

背景（2026-08-16 用户决策）：A 层不再用官方权重——所有方法统一用 splits_v2 train
重训的权重评估，与 B 层唯一差别是评估集。官方权重数字仅作管线验证备案，不进表。

- native = grnade_out_v2/{stem}.native（manifest 口径，289 链一致）
- 口径：池内 best + t=0.1 真均值（mean-of-pool）；NA-MPNN 缺 C1' 链按有效位；AlignIF '-' 跳过
- 界面：eval_inputs_v2 + rcsb_pdb 残基映射（57 号同逻辑），按子集分组
输入：paco_av2_pool / rhodesign_out_av2_retrained / grnade_out_av2_retrained /
      nampnn_out_av2_retrained / alignif_out_av2_retrained
输出：survey/alayer_v2_retrained.json（三视图）+ 打印三张表
"""
import os, sys, json, difflib
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


def interface_positions(pdb_id, chain, native):
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


def nampnn_valid_positions(stem, native):
    os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
    sys.path.insert(0, os.getcwd())
    from data_utils import parse_PDB
    restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
                'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
                'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
    r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
           'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
           'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
           'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
    alph = [r31[r] for r in restypes]
    RM = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}
    d, _, _, _, _ = parse_PDB(os.path.join(SUR, "grnade_inputs_v2", f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    parsed = "".join(RM.get(alph[AA], "?") for AA in d["S"].cpu().numpy())  # 修复：变量名 alph（原 alphabet 未定义）
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
    return pos, len(parsed)


def recs_of(seqs, native, skip_dash=False, nampnn=False, stem=None, t01_n=None):
    """返回 (best, t01_mean, n_used, iface_best, overall_best) 或 None

    t01_mean = 前 t01_n 条(= t=0.1 段，池按温度 0.1/0.5/1.0 顺序拼接)的均值；
    2026-08-19 修复：曾对三温度全池(300 条)求均值却标成 "t=0.1"，
    PACo/RhoDesign/gRNAde 的 t01 列因此偏低 0.5-1.7pp。t01_n=None 表示单温度池(整池)。
    """
    pos_iface = getattr(recs_of, "_iface_pos", None)
    if nampnn:
        full = [s for s in seqs if len(s) == len(native)]
        if not full:
            vpos, plen = nampnn_valid_positions(stem, native)
            sub = [s for s in seqs if len(s) == len(vpos)]
            if not sub:
                return None
            idx = {i: k for k, i in enumerate(vpos)}
            recs = [sum(1 for i in vpos if s[idx[i]] == native[i]) / len(vpos) for s in sub]
            if pos_iface:
                pv = sorted(set(pos_iface) & set(vpos))
                if not pv:
                    return (max(recs), float(np.mean(recs)), len(sub), None, None)
                ib = max(sum(1 for i in pv if s[idx[i]] == native[i]) / len(pv) for s in sub)
                return (max(recs), float(np.mean(recs)), len(sub), ib, max(recs))
            return (max(recs), float(np.mean(recs)), len(sub), None, None)
        seqs = full
    else:
        seqs = [s for s in seqs if len(s) == len(native)]
        if not seqs:
            return None
    recs = []
    for s in seqs:
        if skip_dash:
            den = sum(1 for a in s if a != '-')
            if not den:
                continue
            recs.append(sum(1 for a, b in zip(s, native) if a != '-' and a == b) / den)
        else:
            recs.append(sum(1 for a, b in zip(s, native) if a == b) / len(native))
    if not recs:
        return None
    t01 = float(np.mean(recs[:t01_n])) if t01_n and len(recs) >= t01_n else float(np.mean(recs))
    ib = ov = None
    if pos_iface:
        ipos = sorted(pos_iface)

        def irec(s):
            den = sum(1 for i in ipos if s[i] != '-') if skip_dash else len(ipos)
            if not den:
                return None
            return sum(1 for i in ipos if (not skip_dash or s[i] != '-') and s[i] == native[i]) / den
        if skip_dash:
            ivals = [irec(s) for s in seqs if len(s) == len(native)]
            ivals = [x for x in ivals if x is not None]
            ib = max(ivals) if ivals else None
        else:
            ib = max(irec(s) for s in seqs)
        ov = max(recs)
    return (max(recs), t01, len(recs), ib, ov)


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    indep = sorted(sp["independent_test"])
    zs = sorted(sp["zero_shot_targets"])
    union = sorted(set(indep) | set(zs))

    methods = {
        "PACoAptamer": {"kind": "paco"},
        "RhoDesign": {"kind": "temps", "dir": "rhodesign_out_av2_retrained"},
        "gRNAde": {"kind": "temps", "dir": "grnade_out_av2_retrained"},
        "NA-MPNN": {"kind": "nampnn", "dir": "nampnn_out_av2_retrained"},
        "AlignIF": {"kind": "alignif", "dir": "alignif_out_av2_retrained"},
    }
    per_chain = {}
    for mname, cfg in methods.items():
        for stem in union:
            native_p = os.path.join(SUR, "grnade_out_v2", f"{stem}.native")
            if not os.path.exists(native_p):
                continue
            native = open(native_p).read().strip()
            pdb, ch = stem.rsplit("_", 1)
            pos = interface_positions(pdb, ch, native)
            recs_of._iface_pos = pos
            if cfg["kind"] == "paco":
                # 主配置 ens2-noiface 的 A 层池（2026-08-19 起为默认，不再依赖 env 覆盖）
                pf = os.path.join(os.environ.get(
                    "PACO_AV2_POOL", os.path.join(SUR, "paco_av2_pool_ens2noiface")),
                    f"{stem}.json")
                if not os.path.exists(pf):
                    continue
                d = json.load(open(pf))
                pool = d["pool"]
                assert d["native"] == native
                r = recs_of(pool, native, t01_n=100)
            elif cfg["kind"] == "temps":
                pool = []
                for t in TEMPS:
                    fp = os.path.join(SUR, cfg["dir"], f"{stem}.{t}.seqs")
                    if os.path.exists(fp):
                        pool += [l.strip() for l in open(fp) if l.strip()]
                r = recs_of(pool, native, t01_n=100)
            elif cfg["kind"] == "nampnn":
                fp = os.path.join(SUR, cfg["dir"], f"{stem}.seqs")
                if not os.path.exists(fp):
                    continue
                seqs = [l.strip() for l in open(fp) if l.strip()]
                r = recs_of(seqs, native, nampnn=True, stem=stem)
            else:  # alignif
                faf = os.path.join(SUR, cfg["dir"], stem, "out.fasta")
                if not os.path.exists(faf):
                    continue
                lines = [l.strip() for l in open(faf) if l.strip()]
                seqs = [lines[j + 1] for j, l in enumerate(lines)
                        if l.startswith(">id:") and j + 1 < len(lines)
                        and all(c in "ACGU-" for c in lines[j + 1])]
                r = recs_of(seqs, native, skip_dash=True)
            if r is None:
                continue
            per_chain.setdefault(stem, {})[mname] = {
                "recovery_best": r[0], "recovery_t01_mean": r[1], "n_samples": r[2],
                "iface_best": r[3], "overall_best": r[4],
            }

    def table(subset, name):
        print(f"\n### A 层 v2（重训权重）· {name}（{len(subset)} 链）\n")
        print("| 方法 | 池内 best | t=0.1 均值 | 界面恢复率 | 界面-整体 | 覆盖 | 样本池 |")
        print("|---|---|---|---|---|---|---|")
        rows = {}
        pools = {"PACoAptamer": "3温度×100", "RhoDesign": "3温度×100", "gRNAde": "3温度×100",
                 "NA-MPNN": "t=0.1×10", "AlignIF": "t=0.1×20"}
        for mname in ("PACoAptamer", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"):
            vals = [per_chain[c][mname] for c in subset if c in per_chain and mname in per_chain[c]]
            if not vals:
                continue
            best = float(np.mean([v["recovery_best"] for v in vals]))
            t01 = float(np.mean([v["recovery_t01_mean"] for v in vals]))
            iv = [v for v in vals if v["iface_best"] is not None]
            ir = float(np.mean([v["iface_best"] for v in iv])) if iv else None
            io = float(np.mean([v["overall_best"] for v in iv])) if iv else None
            diff = (ir - io) if iv else None
            rows[mname] = {"best": best, "t01": t01, "iface": ir, "iface_overall": io,
                           "n": len(vals), "n_iface": len(iv)}
            print(f"| {mname} | {100*best:.1f}% | {100*t01:.1f}% | "
                  f"{(f'{100*ir:.1f}%（n='+str(len(iv))+'）') if ir is not None else '—'} | "
                  f"{(f'{100*diff:+.1f}pp') if diff is not None else '—'} | "
                  f"{len(vals)}/{len(subset)} | {pools[mname]} |")
        print("| 随机基线 | 25% | 25% | — | — | — | — |")
        return rows

    out = {"independent_test": table(indep, "independent_test"),
           "zero_shot_targets": table(zs, "zero_shot_targets"),
           "union": table(union, "合并 indep∪zs")}
    json.dump({"views": out, "per_chain": per_chain},
              open(os.path.join(SUR, "alayer_v2_retrained.json"), "w"), indent=1, ensure_ascii=False)
    print("\n写入 survey/alayer_v2_retrained.json", flush=True)


if __name__ == "__main__":
    main()
