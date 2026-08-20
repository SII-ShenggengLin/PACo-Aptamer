"""TACoAptamer 工具函数（复制自 RhoDesign util.py，MIT）"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def nan_to_num(t, nan=0.0, posinf=1e5, neginf=-1e5):
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)


def rbf(values, v_min, v_max, n_bins=16):
    """Returns RBF encodings in a new dimension at the end."""
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    # 修复：RhoDesign 原版对元素 clamp(min=eps) 会把全负差向量压成 1e-8 → 范数极小 → 除法爆炸。
    # 正确做法：clamp 范数本身（只防 0 除，不扭曲方向）
    n = torch.norm(tensor, dim=dim, keepdim=keepdim)
    return torch.clamp(n, min=eps, max=1e5)


def normalize(tensor, dim=-1):
    """Normalizes a tensor along a dimension after removing nans."""
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    Args:
        v: 3D vector, tensor of shape (batch x length x channels x 3)
        R: rotation matrix, tensor of shape (batch x length x 3 x 3)
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Local rotation frame defined by the first three atoms of each residue.
    Args:
        coords: coordinates, tensor of shape (batch_size x length x n_atoms x 3)
    Returns:
        Local rotation frames (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R
