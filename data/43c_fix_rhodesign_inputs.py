#!/usr/bin/env python3
"""第 43c 步：RhoDesign B 层 v2 4 条 FAIL 链输入修复

背景（2026-08-15 诊断）：
- 29v2 的 4 条链（1EI2_A/1O15_A/1OLN_C/2KGP_A）为 s2 独有链，不在 chain_pdb 旧目录，
  走 chain_pdb_v2 兜底 → 该目录文件坐标列偏移（'0  1.0'）→ biotite load 失败
- grnade_inputs_btest_v2（43b 干净管线）的对应文件为多拷贝未去重（1EI2 17 拷贝/2KGP 10 模型）
  → extract 序列 425/250nt，与 manifest 不符
- 统一模式：提取序列 = 5' 端多 1 个 G + manifest + 3' 端配体残基（NMY/MIX/TEP）

修复：从 rcsb_pdb 提取首个拷贝 → 用 manifest 序列对齐定位 → 保留 manifest 对应残基 → 写干净单拷贝 PDB。
输出：survey/rhodesign_btest_fix4/{stem}.pdb（29v2 的 FIX4_DIR 优先路径）
校验：biotite 重新提取的序列 == manifest
"""
import os, json

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
RCSB = os.path.join(DATA_DIR, "rcsb_pdb")
OUT = os.path.join(DATA_DIR, "survey", "rhodesign_btest_fix4")
os.makedirs(OUT, exist_ok=True)

# 目标链：chain_pdb_v2 全部 11 条 s2 独有链（均属 B 层 test；1EI2/2KGP 多拷贝，其余列偏移）
# 2026-08-15 晚扩展：原 4 条修复后日志又现 6IZP_A/6OD9_A/6OD9_B/6UET_A/7AAV_Z/7ABI_Z 共 10 条 FAIL
STEMS = ["1EI2_A", "1O15_A", "1OLN_C", "2KGP_A", "6IZP_A", "6OD9_A", "6OD9_B",
         "6UET_A", "7AAV_Z", "7ABI_Z", "8G8Z_R"]

MOD_MAP = {
    "1MA": "A", "MIA": "A", "6MA": "A", "2MA": "A", "M2A": "A", "M6A": "A",
    "M7A": "A", "MMU": "A", "5MC": "C", "M5C": "C", "CMO": "C", "2MG": "G",
    "7MG": "G", "M2G": "G", "G7M": "G", "MAG": "G", "PSU": "U", "5MU": "U",
    "H2U": "U", "OMU": "U", "DHU": "U", "S4U": "U", "M3U": "U", "I": "U",
    "UNK": "U",
}
STD = {"A", "C", "G", "U"}


def base_of(rn):
    return rn if rn in STD else MOD_MAP.get(rn)


def main():
    manifest = json.load(open(os.path.join(DATA_DIR, "survey", "eval_manifest_v2.json")))
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    seq_map = {}
    for e in entries:
        for ch, s in e.get("sequences", {}).items():
            seq_map[f"{e['pdb']}_{ch}"] = s

    ok = 0
    for stem in STEMS:
        pdb_id, chain = stem.split("_")
        target = seq_map[stem]

        # 1) 原始 PDB 行，chain 过滤 + (resnum, icode) 首见去重（保序）
        seen, kept = set(), []
        with open(os.path.join(RCSB, f"{pdb_id}.pdb")) as f:
            for line in f:
                if line[:4] not in ("ATOM", "HETA"):
                    continue
                if line[21] != chain or line[16] not in (" ", "A"):
                    continue
                key = (line[22:26].strip(), line[26].strip())
                if key in seen:
                    continue
                seen.add(key)
                kept.append(line)

        # 2) 残基分组（保序）：残基名 → 碱基
        groups = []  # [(resid_line_first, seq_char)]
        cur = None
        for line in kept:
            rn = line[17:20].strip()
            b = base_of(rn)
            if b is None:
                continue  # 非 RNA（配体/离子）
            key = (line[22:26].strip(), line[26].strip())
            if cur is None or cur[0] != key:
                cur = (key, [])
                groups.append(cur)
            cur[1].append(line)
        seq = "".join(base_of(line[17:20].strip()) for _, ls in groups for line in ls)

        # 3) 用 manifest 对齐定位（容忍 5' 端偏移 / 3' 端配体混入 / 中间单碱基冗余）
        start, n = None, None
        skip = None  # 需删除的残基下标（单碱基冗余，如 6OD9_B 第 44 位 U）
        if seq == target:
            start, n = 0, len(target)
        elif seq.startswith("G" + target):
            start, n = 1, len(target)
        elif target in seq:
            start = seq.index(target)
            n = len(target)
        else:
            # 单碱基删除对齐：枚举删除位置，若删除后 == target → 该位冗余
            for i in range(len(seq)):
                if seq[:i] + seq[i + 1:] == target:
                    start, n, skip = 0, len(target) + 1, i
                    break
        if start is None:
            print(f"[FAIL] {stem}: 提取 {seq} 无法对齐 manifest {target}")
            continue

        out_lines = []
        for i, (_, ls) in enumerate(groups):
            if start <= i < start + n:
                if skip is not None and i == skip:
                    continue  # 删除冗余残基（6OD9_B 型）
                out_lines.extend(ls)
        out_lines.append("END\n")
        out_path = os.path.join(OUT, f"{stem}.pdb")
        with open(out_path, "w") as f:
            f.writelines(out_lines)

        # 4) 校验：biotite 提取序列 == manifest
        import biotite.structure.io.pdb as bpdb
        import biotite.structure as struc
        arr = bpdb.get_structure(bpdb.PDBFile.read(out_path), model=1)
        got = "".join(struc.get_residues(arr)[1])
        got = "".join(base_of(rn) for rn in got)
        status = "✅" if got == target else f"⚠️ 校验 {got}"
        print(f"{stem}: 对齐成功 ({len(target)}nt, 提取 {n} 残基) {status}")
        ok += 1

    print(f"完成: {ok}/{len(STEMS)}")


if __name__ == "__main__":
    main()
