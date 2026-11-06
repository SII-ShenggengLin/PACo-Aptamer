"""PACoAptamer 数据工具：S1/S2 样本加载、特征化、数据增强

- 坐标：gRNAde 27 原子 → PACo 9 原子 [C4', C1', C2', C3', N1, C2, C5', O5', P]
  （C1' 索引 1 = GVP 中心；前 3 原子含糖环褶皱信息）
- SS：dot-bracket → 配对矩阵 (L, L)
- 配体：元素 one-hot(5) + 到最近 RNA C1' 距离 RBF(16) → (n_lig, 21)
- 增强（训练时）：坐标高斯噪声 σ=0.1Å + 随机旋转（SE(3) 增强，防记忆）
"""
import os
import re
import random
import numpy as np
import torch

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"

# gRNAde 27 原子顺序（processed.pt / s2_samples.pt 的 coords_27 索引）
G27 = ["P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
       "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
       "N6", "N7", "C8", "N9", "OP1", "OP2"]
# PACo 9 原子顺序：前 7 与 RhoDesign 完全一致（前 3 = C4', C1', N1 编码碱基朝向，
# GVP node features 只用前 3 + C1' 中心——碱基方向对序列恢复是关键信息！）
# 后 2 个（C2', C3'）预留糖环褶皱特征（M1 第二版接入）
PACO9 = ["C4'", "C1'", "N1", "C2", "C5'", "O5'", "P", "C2'", "C3'"]
PACO9_IDX = [G27.index(a) for a in PACO9]

ELEMENT_ORDER = ["C", "N", "O", "P", "S"]

FILL = 1e-5


def to_paco9(coords_27):
    """(L, 27, 3) → (L, 9, 3)
    关键：gRNAde 缺失原子填 FILL=1e-5（会被 GVP 当真实坐标 → 特征污染！）
    → 整原子行 ≈ FILL 的位置转为 nan（走 GVP 的 coord_mask 缺失路径，与 RhoDesign biotite 一致）
    """
    c = coords_27[:, PACO9_IDX, :].astype(np.float32).copy()
    fill_rows = (np.abs(c - FILL).max(axis=-1) < 1e-4)
    c[fill_rows] = np.nan
    return c


def ss_to_pairs(ss):
    """dot-bracket → (L, L) 配对矩阵"""
    L = len(ss)
    pairs = np.zeros((L, L), dtype=bool)
    stack = []
    for i, c in enumerate(ss):
        if c in "([{<":
            stack.append(i)
        elif c in ")]}>":
            if stack:
                j = stack.pop()
                pairs[i, j] = pairs[j, i] = True
    return pairs


def element_onehot(elem):
    v = np.zeros(5, dtype=np.float32)
    if elem in ELEMENT_ORDER:
        v[ELEMENT_ORDER.index(elem)] = 1.0
    else:
        v[-1] = 1.0   # 其他
    return v


def lig_features(lig_coords, lig_elements, rna_c1p):
    """配体特征 (n_lig, 21)：元素 one-hot(5) + 到最近 RNA C1' 距离 RBF(16)
    rna_c1p: (L, 3) 所有残基 C1' 坐标
    """
    if len(lig_coords) == 0:
        return np.zeros((0, 21), dtype=np.float32)
    feats = np.stack([element_onehot(e) for e in lig_elements])
    # 距离 RBF
    d = np.linalg.norm(lig_coords[:, None, :] - rna_c1p[None, :, :], axis=-1).min(axis=-1)
    d = np.clip(d, 0.0, 20.0)
    mu = np.linspace(0.0, 20.0, 16)
    sigma = 20.0 / 16
    rbf = np.exp(-((d[:, None] - mu[None, :]) ** 2) / (2 * sigma ** 2))
    return np.concatenate([feats, rbf], axis=-1).astype(np.float32)


def load_s1(chain_subset=None, cache=None):
    """S1 数据：processed.pt（blayer 划分链）→ {stem: (coords9, ss_pairs, seq)}"""
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    data = torch.load(os.path.join(DATA_DIR, "grnade_btrain", "processed.pt"),
                      map_location="cpu", weights_only=False)
    out = {}
    for stem, d in data.items():
        if chain_subset is not None and stem not in chain_subset:
            continue
        seq = d["sequence"]
        coords27 = np.array(d["coords_list"][0], dtype=np.float32)   # (L, 27, 3)
        if coords27.shape[0] != len(seq):
            continue
        ss = d["sec_struct_list"][0] if "sec_struct_list" in d else None
        if ss is None or len(ss) != len(seq):
            continue
        out[stem] = {
            "coords9": to_paco9(coords27),
            "ss_pairs": ss_to_pairs(ss),
            "seq": seq,
        }
    if cache:
        torch.save(out, cache)
    return out


def load_s2(cache=None, src="s2_samples.pt"):
    """S2 数据：s2_samples.pt → {stem: {coords9, ss_pairs, seq, lig_feats, interface}}
    src 可指定干净版 s2_samples_clean.pt（泄漏剔除后）"""
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    data = torch.load(os.path.join(DATA_DIR, "s2_btrain", src),
                      map_location="cpu", weights_only=False)
    out = {}
    for stem, d in data.items():
        coords27 = np.array(d["coords_27"], dtype=np.float32)
        if coords27.shape[0] != len(d["sequence"]):
            continue
        ss = d["ss"]
        if len(ss) != len(d["sequence"]):
            continue
        rna_c1p = coords27[:, G27.index("C1'"), :]
        lig_c = np.array(d["ligand_coords"], dtype=np.float32)
        lig_e = d["ligand_elements"]
        lf = lig_features(lig_c, lig_e, rna_c1p)
        if lf.shape[0] == 0:
            continue
        out[stem] = {
            "coords9": to_paco9(coords27),
            "ss_pairs": ss_to_pairs(ss),
            "seq": d["sequence"],
            "lig_feats": lf,
            "interface": np.array(d["interface_mask"], dtype=bool),
        }
    if cache:
        torch.save(out, cache)
    return out


def augment(coords9, noise_std=0.1):
    """坐标高斯噪声 + 随机旋转（SE(3) 增强）"""
    coords = coords9.copy()
    L = coords.shape[0]
    # 旋转
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(0, np.pi)
    psi = random.uniform(0, 2 * np.pi)
    Rz1 = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    Rz2 = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R = Rz2 @ Rx @ Rz1
    coords = coords @ R.T
    # 噪声（只对有效坐标；FILL 位置不动）
    valid = np.all(np.isfinite(coords), axis=-1)
    noise = np.random.normal(0, noise_std, coords.shape).astype(np.float32)
    coords = np.where(valid[..., None], coords + noise, coords)
    return coords.astype(np.float32)
