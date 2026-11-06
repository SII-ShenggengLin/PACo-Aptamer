#!/usr/bin/env python3
"""第 52 步：NA-MPNN v2 重训数据预处理（splits_v2 train/val × canonical_v2 链库）

为官方 na_run.py 训练管线准备：
1. 每条链的 8 个预处理 npy（asmb_lengths + 7 个 asmb_* 掩码/索引）——
   用 NA-MPNN 官方 PDBDataset.load_for_structure_preprocessing 解析得到精确残基数
   （掩码长度必须与训练时 loader 的解析结果一致；单链无蛋白 → interface 类掩码全零，
   碱基对掩码仅用于指标/增广（本配置全关），置零不影响训练动态）。
2. train.csv / valid.csv（na_run.py 期望的全部列）。

输入：data/canonical_v2/{stem}.pdb（51 号构建）+ splits_v2.json
输出：data/nampnn_btrain_v2/preproc/{asmb_lengths,asmb_interface_masks,...}/{stem}.npy
      data/nampnn_btrain_v2/train.csv / valid.csv
      survey/52_nampnn_preproc_dropped.json（解析失败/无有效 RNA 残基的链）
运行：python data/52_build_nampnn_preproc.py --shard i --nshards N（分片并行）
      不带参数 = 全量（串行）
"""
import os, sys, json, argparse
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
NAMPNN_DIR = os.path.join(ROOT, "tools", "NA-MPNN")
CANON = os.path.join(DATA_DIR, "canonical_v2")
OUT_DIR = os.path.join(DATA_DIR, "nampnn_btrain_v2")
PREPROC = os.path.join(OUT_DIR, "preproc")

SUBDIRS = ["asmb_lengths", "asmb_interface_masks", "asmb_side_chain_interface_masks",
           "asmb_nearest_protein_side_chain_index", "asmb_base_pair_masks",
           "asmb_base_pair_index", "asmb_canonical_base_pair_masks",
           "asmb_canonical_base_pair_index", "sequences"]


def build_dataset():
    """与 preprocess_dataset.py 相同的 PDBDataset 构造（ATOMS_TO_LOAD=all）"""
    sys.path.insert(0, NAMPNN_DIR)
    import cifutils, pdbutils
    from na_data_utils import PDBDataset
    params = json.load(open(os.path.join(NAMPNN_DIR, "data", "preprocess_dataset.json")))
    atom_list = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2',
                 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2',
                 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
                 'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
                 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2',
                 'N2', 'O2', 'N1']
    cif_parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"],
                                    randomize_nmr_model=params["RANDOMIZE_NMR_MODEL"])
    pdb_parser = pdbutils.PDBParser()
    return PDBDataset(cif_parser=cif_parser, pdb_parser=pdb_parser,
                      atom_list_to_save=atom_list,
                      parse_protein=params["PARSE_PROTEIN"], parse_dna=params["PARSE_DNA"],
                      parse_rna=params["PARSE_RNA"], parse_rna_as_dna=params["PARSE_RNA_AS_DNA"],
                      na_shared_tokens=params["NA_SHARED_TOKENS"],
                      protein_backbone_occ_cutoff=params["PROTEIN_BACKBONE_OCC_CUTOFF"],
                      protein_side_chain_occ_cutoff=params["PROTEIN_SIDE_CHAIN_OCC_CUTOFF"],
                      dna_backbone_occ_cutoff=params["DNA_BACKBONE_OCC_CUTOFF"],
                      dna_side_chain_occ_cutoff=params["DNA_SIDE_CHAIN_OCC_CUTOFF"],
                      rna_backbone_occ_cutoff=params["RNA_BACKBONE_OCC_CUTOFF"],
                      rna_side_chain_occ_cutoff=params["RNA_SIDE_CHAIN_OCC_CUTOFF"],
                      crop_large_structures=params["CROP_LARGE_STRUCTURES"],
                      batch_tokens=params["BATCH_TOKENS"],
                      na_ref_atom=params["NA_REF_ATOM"])


def process_one(ds, stem):
    """返回 (status, detail)；成功时写 8 个 npy + sequences csv"""
    path = os.path.join(CANON, f"{stem}.pdb")
    if not os.path.exists(path):
        return "no_canonical_pdb", None
    if os.path.exists(os.path.join(PREPROC, "asmb_lengths", f"{stem}.npy")):
        return "cached", None
    try:
        assemblies, chain_sequences = ds.load_for_structure_preprocessing({"structure_path": path})
    except Exception as e:
        return f"parse_error:{type(e).__name__}", None
    # 失败时官方返回 ("pass", "pass")（字符串元组），成功时为 [(assembly_id, out_dict), ...]
    if not assemblies or isinstance(assemblies[0], str):
        return "parse_pass", None
    amap = dict(assemblies)
    if "1" not in amap:
        return "no_assembly_1", None
    out = amap["1"]
    L = int(out["macromolecule_L"])
    pL, dL, rL = int(out["protein_L"]), int(out["dna_L"]), int(out["rna_L"])
    if L == 0 or rL == 0:
        return "no_valid_rna_residue", None
    base = os.path.join(PREPROC)
    np.save(os.path.join(base, "asmb_lengths", f"{stem}.npy"),
            {"1": (L, pL, dL, rL)})
    np.save(os.path.join(base, "asmb_interface_masks", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int32)})
    np.save(os.path.join(base, "asmb_side_chain_interface_masks", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int32)})
    np.save(os.path.join(base, "asmb_nearest_protein_side_chain_index", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int64)})
    np.save(os.path.join(base, "asmb_base_pair_masks", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int32)})
    np.save(os.path.join(base, "asmb_base_pair_index", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int64)})
    np.save(os.path.join(base, "asmb_canonical_base_pair_masks", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int32)})
    np.save(os.path.join(base, "asmb_canonical_base_pair_index", f"{stem}.npy"),
            {"1": np.zeros(L, dtype=np.int64)})
    seq = "".join(chain_sequences[0][2]) if chain_sequences else ""
    with open(os.path.join(base, "sequences", f"{stem}.csv"), "w") as f:
        f.write("chain_id,chain_type,sequence\n")
        for cid, ctype, s in chain_sequences:
            f.write(f"{cid},{ctype},{s}\n")
    return "ok", L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    for d in SUBDIRS:
        os.makedirs(os.path.join(PREPROC, d), exist_ok=True)

    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    train = sorted(sp["train"])
    val = sorted(sp["val"])
    todo = [("train", c) for c in train] + [("val", c) for c in val]
    todo = [x for i, x in enumerate(todo) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] {len(todo)} 条", flush=True)

    ds = build_dataset()
    stats = {}
    dropped = {}
    for i, (split, stem) in enumerate(todo):
        status, detail = process_one(ds, stem)
        stats[status] = stats.get(status, 0) + 1
        if status != "ok":
            dropped[stem] = status
        if (i + 1) % 200 == 0:
            print(f"  进度 {i+1}/{len(todo)} ok={stats.get('ok',0)} drop={len(dropped)}", flush=True)
    os.makedirs(os.path.join(DATA_DIR, "survey"), exist_ok=True)
    json.dump(dropped, open(os.path.join(
        DATA_DIR, "survey", f"52_preproc_dropped_shard{args.shard}.json"), "w"), indent=1)
    print(f"[shard {args.shard}] 完成: {stats}", flush=True)


if __name__ == "__main__":
    main()
