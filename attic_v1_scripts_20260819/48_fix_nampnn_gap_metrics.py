#!/usr/bin/env python3
"""第 48 步：NA-MPNN 缺坐标残基链的评估对齐修复（数据质量问题，非代码 bug）

背景（v2 A 层审计，2026-08-16）：
- NA-MPNN 推理解析器（inference/data_utils.py parse_PDB）以 C1' 为参考原子建图，
  原始 PDB 中缺 C1' 的残基被直接丢弃 → 输出序列短于 native 且无位置标记。
- 5 条 A 层链受此影响（输出长度 < native）：5NFV_B(39/41) 7E9I_A(32/33)
  8U5J_A(35/36) 8VPV_A(98/99) 9GUT_X(12/17)，32v2 按长度过滤整链丢弃（284/289）。
- 这些缺坐标位点无法补数据（坐标本身不存在），处理方式与 AlignIF 的 '-' gap
  口径一致：只在 NA-MPNN 实际解析到的（有坐标的）位置上评估恢复率。

做法：
1. 复现 NA-MPNN 官方解析（同一 parse_PDB 函数、同一输入 PDB）→ 解析序列；
2. 解析序列是 native 的子序列（缺坐标位删除）→ difflib 对齐得到有效位置集合；
3. 恢复率 / 界面恢复率只在有效位置上计算（界面 = 界面位 ∩ 有效位）；
4. 结果并入 metrics_NAMPNN_v2.json（coverage 289/289）+ interface_metrics_v2.json。

产物：survey/nampnn_valid_positions_v2.json（每链有效位置，供 B 层指标脚本复用逻辑）
"""
import os, sys, json, difflib
from collections import defaultdict

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
NP_OUT = os.path.join(DATA_DIR, "survey", "nampnn_out_v2")
GR_OUT = os.path.join(DATA_DIR, "survey", "grnade_out_v2")
EI = os.path.join(DATA_DIR, "survey", "eval_inputs_v2")
PDB_IN = os.path.join(DATA_DIR, "survey", "grnade_inputs_v2")

# NA-MPNN 字母表（inference/run.py 同款，na_shared_tokens=0 口径）
restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
            'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
            'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
       'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
alphabet = [r31[r] for r in restypes]
RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}


def parse_native_positions(stem):
    """用 NA-MPNN 官方 parse_PDB 复现解析，返回 (解析序列, 有效位置列表)"""
    os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
    sys.path.insert(0, os.getcwd())
    from data_utils import parse_PDB  # 延迟 import（依赖 prody）
    d, _, _, _, _ = parse_PDB(os.path.join(PDB_IN, f"{stem}.pdb"), device="cpu",
                              model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    S = d["S"].cpu().numpy()
    parsed = "".join(RNA_MAP.get(alphabet[AA], "?") for AA in S)
    return parsed


def align_positions(parsed, native):
    """parsed 是 native 删除若干位后的子序列 → 返回 native 中被保留的位置下标"""
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
        elif tag in ("delete", "replace"):
            pass  # native 有而 parsed 无 → 缺坐标位，丢弃
        # insert（parsed 有而 native 无）不应出现
    assert "".join(native[i] for i in pos) == parsed, f"对齐失败: {parsed} vs 子序列不符"
    return pos


def recovery_pos(seq, native, pos):
    return sum(1 for i in pos if seq[k] == native[i] for k, i in enumerate(pos)) / max(1, len(pos))


# 32v2 同款 MOD_MAP / interface_positions（保持口径一致）
MOD_MAP = json.load(open(os.path.join(DATA_DIR, "survey", "mod_map.json"))) if os.path.exists(os.path.join(DATA_DIR, "survey", "mod_map.json")) else None
if MOD_MAP is None:
    # 从 32v2 脚本静态导入（避免复制 130 行表）
    import importlib.util
    spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
    m32 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m32)
    MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}


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
        return set()
    e = json.load(open(os.path.join(EI, f"{pdb_id}.json")))
    iface_keys = set()
    for lc in e.get("ligand_contacts", []):
        for ch, keys in lc.get("interface_chains", {}).items():
            if ch == chain:
                for (rn_num, ic) in keys:
                    iface_keys.add((rn_num, ic))
    positions = set()
    for i, (rn, rn_num, ic) in enumerate(seq_keys):
        if (rn_num, ic) in iface_keys:
            positions.add(i)
    return positions


def main():
    m_path = os.path.join(DATA_DIR, "survey", "metrics_NAMPNN_v2.json")
    m = json.load(open(m_path))
    kept = m["per_chain"]
    all_stems = sorted(f[:-5] for f in os.listdir(NP_OUT) if f.endswith(".seqs"))
    dropped = [s for s in all_stems if s not in kept]
    print(f"待修复链: {dropped}", flush=True)
    if not dropped:
        print("无需修复", flush=True)
        return

    valid_pos_cache = {}
    iface_new = {}
    for stem in dropped:
        native = open(os.path.join(GR_OUT, f"{stem}.native")).read().strip()
        seqs = [l.strip() for l in open(os.path.join(NP_OUT, f"{stem}.seqs")) if l.strip()]
        parsed = parse_native_positions(stem)
        pos = align_positions(parsed, native)
        valid_pos_cache[stem] = pos
        seqs = [s for s in seqs if len(s) == len(parsed)]
        assert seqs, f"{stem} 无有效输出"
        # 恢复率仅在有效位置上
        recs = [sum(1 for k, i in enumerate(pos) if s[k] == native[i]) / len(pos) for s in seqs]
        kept[stem] = {
            "stem": stem, "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": sum(recs) / len(recs),
            "n_samples": len(seqs),
            "n_valid_pos": len(pos),  # 缺坐标位标记（评估只在这些位上）
        }
        print(f"  {stem}: native {len(native)}nt, 有效位 {len(pos)}, best {max(recs):.3f}", flush=True)
        # 界面（有效位 ∩ 界面位）
        pdb, ch = stem.rsplit("_", 1)
        ipos_full = interface_positions(pdb, ch, native)
        ipos = sorted(i for i in ipos_full if i in set(pos))
        if ipos:
            pos_index = {i: k for k, i in enumerate(pos)}
            def iface_rec(s):
                return sum(1 for i in ipos if s[pos_index[i]] == native[i]) / len(ipos)
            best_iface = max(iface_rec(s) for s in seqs)
            best_overall = max(recs)
            iface_new[stem] = {"NAMPNN": {"iface_t0.1": best_iface, "overall_t0.1": best_overall}}

    # 汇总重算
    recs_best = [r["recovery_best"] for r in kept.values()]
    recs_mean = [r["recovery_t01_mean"] for r in kept.values()]
    m["summary"]["n_chains"] = len(kept)
    m["summary"]["coverage"] = f"{len(kept)}/289"
    m["summary"]["recovery_best_mean"] = sum(recs_best) / len(recs_best)
    m["summary"]["recovery_best_median"] = sorted(recs_best)[len(recs_best) // 2]
    m["summary"]["recovery_t01_mean_mean"] = sum(recs_mean) / len(recs_mean)
    m["summary"]["gap_fix_note"] = ("5 链（5NFV_B/7E9I_A/8U5J_A/8VPV_A/9GUT_X）原始 PDB 缺 C1' 坐标，"
                                    "NA-MPNN 解析丢弃这些残基；按 AlignIF gap 口径仅在有效位上评估")
    json.dump(m, open(m_path, "w"), indent=1, ensure_ascii=False)

    # 界面并入（重算 NA-MPNN 界面汇总）
    im_path = os.path.join(DATA_DIR, "survey", "interface_metrics_v2.json")
    if os.path.exists(im_path):
        im = json.load(open(im_path))
        per = im.get("per_chain", {})
        per.update(iface_new)
        na_if = [v["NAMPNN"]["iface_t0.1"] for v in per.values() if "NAMPNN" in v]
        na_ov = [v["NAMPNN"]["overall_t0.1"] for v in per.values() if "NAMPNN" in v]
        if na_if:
            im["per_chain"] = per
            im.setdefault("summary", {})["NAMPNN"] = {
                "n_chains": len(na_if),
                "interface_recovery_t01_best": sum(na_if) / len(na_if),
                "overall_recovery_t01_best": sum(na_ov) / len(na_ov),
            }
            json.dump(im, open(im_path, "w"), indent=1, ensure_ascii=False)
        m["summary"]["interface_recovery_t01_best"] = sum(na_if) / len(na_if)
        m["summary"]["overall_recovery_t01_best"] = sum(na_ov) / len(na_ov)
        m["summary"]["n_iface_chains"] = len(na_if)
        json.dump(m, open(m_path, "w"), indent=1, ensure_ascii=False)

    json.dump(valid_pos_cache,
              open(os.path.join(DATA_DIR, "survey", "nampnn_valid_positions_v2.json"), "w"), indent=1)
    print(f"修复完成: coverage {m['summary']['coverage']}, "
          f"best {m['summary']['recovery_best_mean']:.4f}, "
          f"t01 {m['summary']['recovery_t01_mean_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
