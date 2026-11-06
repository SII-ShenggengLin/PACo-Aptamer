#!/usr/bin/env python3
"""第 53 步：AlignIF v2 重训数据构建（splits_v2 train/val × canonical_v2 链库）

输出 AlignIF 官方 processed 格式（已逐字段对照官方 data/alignif/data/processed/*.pt 验证）：
  {stem}.pt = {
    seq: str（字母 A/C/G/U/N；'N' = 未指明碱基残基，AlignIF loss ignore_index=4 忽略）,
    coords: (L, 14, 3) float32，缺失原子 = NaN，
            列序 BB_ATOMS = [P,O5',C5',C4',O4',C3',O3',C2',O2',C1',N,C,C2,N1]，
            N = N9(A/G)/N1(C/U)，C = C4(A/G)/C2(C/U)，C2/N1 列 = C/N 列复制（官方惯例）,
    chain_ids: list[str]（每残基 'A'）,
    seq_ids: list[int]（PDB 残基号）,
    ss: (L, L) bool 全 False（AlignIFDataset 不读，占位保持 schema 一致）
  }
划分：split.pt = {train, val, test, independent}（test/independent 留空，datamodule 已打补丁跳过空集）
QC 剔除：骨架完整（C1'&C4'）残基 < 50% 或 <5nt 的链（含 29 条蛋白痕迹垃圾链）→ 记录
输出：data/alignif_btrain_v2/processed/{stem}.pt + split.pt + survey/53_alignif_data_dropped.json
"""
import os, json
import numpy as np
import torch

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CANON = os.path.join(DATA_DIR, "canonical_v2")
OUT_DIR = os.path.join(DATA_DIR, "alignif_btrain_v2")
PROC = os.path.join(OUT_DIR, "processed")
os.makedirs(PROC, exist_ok=True)

BB_ATOMS = ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N", "C", "C2", "N1"]
N_ATOM = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}   # idx10
C_ATOM = {"A": "C4", "G": "C4", "C": "C2", "U": "C2"}   # idx11


def convert(stem, seq):
    path = os.path.join(CANON, f"{stem}.pdb")
    if not os.path.exists(path):
        return None, "no_canonical_pdb"
    # 解析残基（canonical 文件已去重/model1/altloc 过滤，链名 A）
    residues = {}
    order = []
    resnums = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
                resnums.append(int(line[22:26].strip()))
            residues[key][atom] = xyz
    L = len(order)
    if L != len(seq):
        return None, f"len_mismatch pdb={L} seq={len(seq)}"
    coords = np.full((L, 14, 3), np.nan, dtype=np.float32)
    n_bb_ok = 0
    for i, key in enumerate(order):
        base = seq[i]
        for j, atom in enumerate(BB_ATOMS):
            want = atom
            if atom == "N":
                want = N_ATOM.get(base)
            elif atom == "C":
                want = C_ATOM.get(base)
            elif atom == "C2":
                want = C_ATOM.get(base)   # 官方惯例：C2 列 = C 列复制
            elif atom == "N1":
                want = N_ATOM.get(base)   # N1 列 = N 列复制
            if want and want in residues[key]:
                coords[i, j] = residues[key][want]
        if "C1'" in residues[key] and "C4'" in residues[key]:
            n_bb_ok += 1
    if L < 5:
        return None, "too_short"
    if n_bb_ok < L * 0.5:
        return None, f"backbone_incomplete {n_bb_ok}/{L}"
    return {
        "seq": seq,
        "coords": coords,
        "chain_ids": ["A"] * L,
        "seq_ids": resnums,
        "ss": torch.zeros((L, L), dtype=torch.bool),
    }, "ok"


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    meta = json.load(open(os.path.join(DATA_DIR, "survey", "canonical_v2.json")))
    split_ids = {"train": sorted(sp["train"]), "val": sorted(sp["val"])}
    dropped = {}
    for mode, ids in split_ids.items():
        ok = 0
        for i, stem in enumerate(ids):
            entry = meta.get(stem)
            if entry is None:
                dropped[stem] = "no_canonical"
                continue
            data, status = convert(stem, entry["seq"])
            if data is None:
                dropped[stem] = status
                continue
            torch.save(data, os.path.join(PROC, f"{stem}.pt"))
            ok += 1
            if (i + 1) % 1000 == 0:
                print(f"[{mode}] 进度 {i+1}/{len(ids)}", flush=True)
        print(f"[{mode}] 完成 {ok}/{len(ids)}", flush=True)
    # split.pt（test/independent 留空——训练只用 train/val；datamodule 已补丁跳过空集）
    split = {
        "train": [s for s in split_ids["train"] if s not in dropped],
        "val": [s for s in split_ids["val"] if s not in dropped],
        "test": [],
        "independent": [],
    }
    torch.save(split, os.path.join(OUT_DIR, "split.pt"))
    json.dump(dropped, open(os.path.join(DATA_DIR, "survey", "53_alignif_data_dropped.json"), "w"), indent=1)
    print(f"split: train {len(split['train'])} / val {len(split['val'])}; 剔除 {len(dropped)}", flush=True)


if __name__ == "__main__":
    main()
