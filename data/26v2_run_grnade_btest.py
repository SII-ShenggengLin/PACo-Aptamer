#!/usr/bin/env python3
"""第 17 步：gRNAde 批量逆折叠推理（官方权重 × 评估集）

对 grnade_inputs/ 的每个链 PDB：
- 3D 模式特征化（k-NN 图）
- 每个温度（0.1/0.5/1.0）采样 N 个序列 → 每链一个序列文件
输出：survey/grnade_out/{pdb}_{chain}.{temp}.seqs
GPU：1、2（用户指定）
"""
import os, sys, json, yaml, torch
import numpy as np

# gRNAde 导入所需环境变量（推理仅 3D 模式，占位路径即可）
os.environ.setdefault("X3DNA", "/tmp/x3dna")
os.environ.setdefault("ETERNAFOLD", "/tmp/eternafold")
os.environ.setdefault("PROJECT_PATH", "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")
from src.models import gRNAde
from src.data.featurizer import RNAGraphFeaturizer

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
GRNADE_ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde"
IN_DIR = os.environ.get("GR_IN_DIR", os.path.join(DATA_DIR, "survey", "grnade_inputs_btest_v2"))  # 43b 构建（rcsb 切链 + manifest 序列）
CP2_DIR = os.path.join(DATA_DIR, "chain_pdb_v2")
OUT_DIR = os.environ.get("GR_OUT_DIR", os.path.join(DATA_DIR, "survey", "grnade_out_btest_v2"))
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 100          # 每温度采样数
TEMPERATURES = [0.1, 0.5, 1.0]

def samples_for_len(n):
    """按链长自适应采样数（控制显存：长链降采样）"""
    if n <= 200:
        return N_SAMPLES
    if n <= 400:
        return 50
    return 30

def load_model(gpu):
    with open(os.path.join(GRNADE_ROOT, "checkpoints", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model = gRNAde(
        node_in_dim=(15, 4), node_h_dim=(128, 16),
        edge_in_dim=(132, 3), edge_h_dim=(64, 4),
        num_layers=4, drop_rate=0.5, out_dim=4,
    )
    ckpt = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/gRNAde_btrain_v2/best_checkpoint.h5"  # B 层 v2 重训
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(f"cuda:{gpu}")
    model.eval()
    featurizer = RNAGraphFeaturizer(
        split="test", radius=0.0, top_k=32,
        num_rbf=32, num_posenc=32, max_num_conformers=1,
        noise_scale=0.1, drop_prob_3d=0.75,
    )
    return model, featurizer

def seq_to_str(seq_tensor, vocab):
    return "".join(vocab[i] for i in seq_tensor.tolist())

def featurize_custom(featurizer, stem):
    """用我们 RNAfold 的 .ss + 手工提取骨架坐标构造 rna dict
    （绕开 gRNAde 内部 SS 提取——biotite 对 ≤30nt 短链会失败）"""
    import torch as _t
    pdb_path = os.path.join(IN_DIR, stem + ".pdb")
    if not os.path.exists(pdb_path):
        pdb_path = os.path.join(CP2_DIR, stem + ".pdb")  # s2 独有链兜底
    seq_p = os.path.join(IN_DIR, stem + ".seq")
    ss_p = os.path.join(IN_DIR, stem + ".ss")
    # 43b 构建的三件套输入：.seq/.ss 必存在（与 A 层 17v2 同格式）
    if not os.path.exists(seq_p) or not os.path.exists(ss_p):
        raise ValueError(f"{stem}: 43b 输入缺失")
    seq = open(seq_p).read().strip()
    ss = open(ss_p).read().strip()
    # 本项目补丁：canonical 序列含 'N'（未指明碱基残基）时，结构占位用 'A' 喂 featurizer
    # （native 仍写原始序列；指标计算跳过 'N' 位——63 号统一处理）
    seq = seq.replace("N", "A") if "N" in seq else seq
    # 逐残基提取骨架原子（P, C4', N1/N9）
    residues = {}   # (resnum, icode) -> {atom: (x,y,z)}
    order = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key][atom] = (x, y, z)
    if len(order) != len(seq):
        raise ValueError(f"{stem}: PDB 残基数 {len(order)} != 序列 {len(seq)}")
    coords = []
    for i, key in enumerate(order):
        rn = seq[i]
        base_atom = "N1" if rn in "CU" else "N9"
        pts = []
        for need in ["P", "C4'", base_atom]:
            if need in residues[key]:
                pts.append(residues[key][need])
            else:
                # 缺失原子用填充值（对齐 gRNAde 内部 FILL_VALUE 语义）
                pts.append((1e-5, 1e-5, 1e-5))
        coords.append(pts)
    coords_t = _t.tensor(coords, dtype=_t.float32)  # (L, 3, 3)
    rna = {"sequence": seq, "coords_list": [coords_t], "sec_struct_list": [ss]}
    return featurizer.featurize(rna), rna

def main(gpu, task_list):
    model, featurizer = load_model(gpu)
    vocab = ["A", "C", "G", "U"]
    stats = {"ok": 0, "fail": 0}
    for pdb, chain in task_list:
        stem = f"{pdb}_{chain}"
        pdb_path = os.path.join(IN_DIR, stem + ".pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(CP2_DIR, stem + ".pdb")
        outs = [os.path.join(OUT_DIR, f"{stem}.{t}.seqs") for t in TEMPERATURES]
        if all(os.path.exists(o) for o in outs):
            stats["ok"] += 1
            continue
        try:
            batch, raw = featurize_custom(featurizer, stem)
            batch = batch.to(f"cuda:{gpu}")
            # native = 原始 .seq（含 'N' 时保持 'N'，不用占位序列）
            native = open(os.path.join(IN_DIR, stem + ".seq")).read().strip() \
                if os.path.exists(os.path.join(IN_DIR, stem + ".seq")) else raw["sequence"]
        except Exception as e:
            stats["fail"] += 1
            print(f"  [FAIL] {stem} featurize: {e}", flush=True)
            continue
        # 保存天然序列（恢复率计算用）
        with open(os.path.join(OUT_DIR, f"{stem}.native"), "w") as f:
            f.write(native + "\n")
        n_samp = samples_for_len(len(native))
        for temp, out_f in zip(TEMPERATURES, outs):
            try:
                with torch.no_grad():
                    seqs = model.sample(batch, n_samples=n_samp, temperature=temp)
                with open(out_f, "w") as f:
                    for s in seqs:
                        f.write(seq_to_str(s, vocab) + "\n")
            except Exception as e:
                print(f"  [FAIL] {stem} temp={temp}: {e}", flush=True)
                stats["fail"] += 1
                continue
        stats["ok"] += 1
        print(f"  {stem}: ok", flush=True)
    print(f"[GPU{gpu}] 完成: {stats}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--taskfile", type=str, required=True)
    args = parser.parse_args()
    tasks = [tuple(t) for t in json.load(open(args.taskfile))]
    main(args.gpu, tasks)
