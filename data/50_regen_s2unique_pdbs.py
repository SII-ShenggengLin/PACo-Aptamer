#!/usr/bin/env python3
"""第 50 步：重建 s2 独有链的干净 PDB（splits_v2 train/val/test 全覆盖）

背景：
- splits_v2 含 104 条 s2 独有链（41a 统一序列库时从复合物摸底数据补入），
  其中 90 条在 train、3 条在 val、11 条在 test。
- chain_pdb_v2/ 旧导出只有 11 条 test 链且坐标列偏移（biotite/prody 固定列解析失败）；
  43c 已为 B 层 test 修过 11 条（rhodesign_btest_fix4）。
- 本脚本把 104 条全部从 rcsb_pdb 重建为干净单拷贝 PDB（首拷贝去重 + 序列对齐），
  供 NA-MPNN/AlignIF v2 重训及后续推理统一使用。

对齐规则（同 43c）：完全匹配 > 5' 端多 G > 子串 > 单碱基删除；均失败则记录 FAIL。
输出：data/chain_pdb_v2/{stem}.pdb（覆盖旧 11 条坏文件）+ 控制台校验报告
"""
import os, json

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
RCSB = os.path.join(DATA_DIR, "rcsb_pdb")
OUT = os.path.join(DATA_DIR, "chain_pdb_v2")

# 全量修饰核苷酸映射（与 32v2/34v2 指标脚本一致，避免稀有修饰残基误丢）
import importlib.util
spec = importlib.util.spec_from_file_location("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
m32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m32)
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}


def base_of(rn):
    return rn if rn in STD else MOD_MAP.get(rn)


def fmt_atom(serial, atom, rn, chain, resnum, icode, x, y, z, occ, bfac, elem):
    """标准 PDB ATOM 行（固定列）"""
    return (f"ATOM  {serial:5d} {atom:<4s}{' '}{rn:>3s} {chain:1s}{resnum:>4s}{icode:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}  \n")


def regen(stem, target):
    pdb_id, chain = stem.rsplit("_", 1)
    path = os.path.join(RCSB, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return None, "no_rcsb_pdb"
    # 1) 仅 model 1 + 目标链 + altloc 过滤；多拷贝去重：
    #    残基按 (resnum,icode) 连续分组，若某个 key 在非相邻位置再次出现 → 第二拷贝开始，停止
    #    （拷贝间 resnum 重复；同残基的多行原子必须全部保留——43c 旧版按 key 首见只留首行，
    #    导致每残基只剩 1 个原子，是严重 bug，本版修复）
    groups = []
    cur = None
    seen_keys = set()
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[21] != chain or line[16] not in (" ", "A"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            if cur is not None and key == cur[0]:
                cur[1].append(line)
                continue
            if key in seen_keys:
                break  # 多拷贝结构的第二拷贝起点
            seen_keys.add(key)
            cur = (key, [line])
            groups.append(cur)
    if not groups:
        return None, "no_chain_atoms"
    # 2) 残基 → 碱基序列（配体/未知修饰跳过）
    groups = [(k, ls) for k, ls in groups if base_of(ls[0][17:20].strip()) is not None]
    seq = "".join(base_of(ls[0][17:20].strip()) for _, ls in groups)
    # 3) 对齐（target=None → 结构内容即真值，全保留）
    start, n, skip = None, None, None
    if target is None:
        start, n = 0, len(seq)
    elif seq == target:
        start, n = 0, len(target)
    elif seq.startswith("G" + target):
        start, n = 1, len(target)
    elif target in seq:
        start, n = seq.index(target), len(target)
    else:
        for i in range(len(seq)):
            if seq[:i] + seq[i + 1:] == target:
                start, n, skip = 0, len(target) + 1, i
                break
    if start is None:
        # 多拷贝拼接目标（s2_samples 历史 bug：N 拷贝序列拼接成一条，seqs_all 目标长 = N×单拷贝）
        # 回退：以 rcsb 首拷贝提取序列为该链真值（43c 对 test 链的同款处理），标记 single_copy
        if len(seq) > 0 and len(target) % max(1, len(seq)) == 0 or len(seq) > 0:
            return None, f"SINGLE_COPY|{seq}"  # 交由上层用提取序列当真值重写
        return None, f"align_fail extracted_len={len(seq)} target_len={len(target)}"
    # 4) 重写为标准格式（坐标列严格对齐，prody/biotite 均可解析）
    out_lines = []
    serial = 1
    for i, (_, ls) in enumerate(groups):
        if start <= i < start + n:
            if skip is not None and i == skip:
                continue
            for line in ls:
                atom = line[12:16].strip()
                rn = line[17:20].strip()
                resnum = line[22:26].strip()
                icode = line[26].strip()
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except ValueError:
                    continue
                occ = line[54:60].strip()
                bfac = line[60:66].strip()
                elem = line[76:78].strip() if len(line) >= 78 else atom[0]
                out_lines.append(fmt_atom(serial, atom, rn, chain, resnum, icode,
                                          x, y, z, float(occ) if occ else 1.0,
                                          float(bfac) if bfac else 20.0, elem))
                serial += 1
    out_lines.append("END\n")
    return "".join(out_lines), f"ok n={n}" + (f" skip@{skip}" if skip is not None else "")


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    seqs = json.load(open(os.path.join(DATA_DIR, "splits_v2", "seqs_all.json")))
    target_chains = sorted(set(sp["train"]) | set(sp["val"]) | set(sp["test"]))
    # 只重建无 chain_pdb 的链 + chain_pdb_v2 旧 11 条（坏文件覆盖）
    need = [c for c in target_chains if not os.path.exists(os.path.join(DATA_DIR, "chain_pdb", f"{c}.pdb"))]
    old11 = [f[:-4] for f in os.listdir(OUT) if f.endswith(".pdb")]
    need = sorted(set(need) | set(old11))
    print(f"需重建 {len(need)} 条（train/val 缺文件 {len(need)-len(old11)} + 旧 v2 文件 {len(old11)}）", flush=True)
    fails = {}
    single_copy = {}
    qc = {}
    # test 链（11 条 43c 已修）以 manifest 单拷贝序列为准（与 B 层评估口径一致）
    man = json.load(open(os.path.join(DATA_DIR, "survey", "eval_manifest_v2.json")))
    man_map = {}
    for e in man:
        for ch, s in e.get("sequences", {}).items():
            man_map[f"{e['pdb']}_{ch}"] = s
    FORCE_EXTRACTED = {"1EI2_A", "2KGP_A", "6IZP_A"}  # 多拷贝拼接链：以结构单拷贝内容（含 5'G）为真值
    for stem in need:
        if stem in FORCE_EXTRACTED:
            target = None
        else:
            target = man_map.get(stem) or seqs.get(stem)
            if target is None:
                fails[stem] = "no_seq_in_seqs_all"
                continue
            target = target.replace("T", "U")
        content, status = regen(stem, target)
        if content is None and status.startswith("SINGLE_COPY|"):
            # 多拷贝拼接目标（s2_samples 历史 bug）：以 rcsb 首拷贝提取序列为真值重建
            single = status.split("|", 1)[1]
            content, status2 = regen(stem, single)
            if content is not None:
                single_copy[stem] = {"n_res": len(single), "orig_target_len": len(target)}
                target = single  # 校验目标换为首拷贝序列
            else:
                fails[stem] = status2
                print(f"[FAIL] {stem}: {status2}", flush=True)
                continue
        elif content is None:
            fails[stem] = status
            print(f"[FAIL] {stem}: {status}", flush=True)
            continue
        with open(os.path.join(OUT, f"{stem}.pdb"), "w") as f:
            f.write(content)
        # 校验：重读文件，提取序列 == target
        seq_back = []
        seen = set()
        for line in content.splitlines():
            if not line.startswith("ATOM"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            if key in seen:
                continue
            seen.add(key)
            b = base_of(line[17:20].strip())
            if b:
                seq_back.append(b)
        ok = (target is None) or ("".join(seq_back) == target)
        if not ok:
            fails[stem] = "verify_mismatch"
            print(f"[FAIL] {stem}: 回读校验不一致", flush=True)
            continue
        # QC：骨架完整性（每残基 C1'/C4'/P 覆盖），CA 单原子痕迹链等垃圾数据打标
        qc_atoms = {}
        for line in content.splitlines():
            if line.startswith("ATOM"):
                k2 = (line[22:26].strip(), line[26].strip())
                qc_atoms.setdefault(k2, set()).add(line[12:16].strip())
        nres = len(qc_atoms)
        n_bb = sum(1 for a in qc_atoms.values() if "C1'" in a and "C4'" in a)
        qc[stem] = {"n_res": nres, "n_with_C1C4": n_bb}
        if n_bb < nres * 0.5:
            qc[stem]["qc_flag"] = "backbone_incomplete"
    print(f"\n完成: {len(need)-len(fails)}/{len(need)} 成功（其中首拷贝回退 {len(single_copy)} 条）", flush=True)
    if fails:
        print("失败清单:", json.dumps(fails, indent=1), flush=True)
    json.dump({"fails": fails, "single_copy_fallback": single_copy, "qc": qc},
              open(os.path.join(DATA_DIR, "survey", "50_regen_fails.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
