#!/usr/bin/env python3
"""把 cif-only 条目转为 PDB（修复 RCSB 404 占位文件）— 复现修复步骤
用法: python3 07_convert_cif.py
"""
import os, subprocess

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA, "rcsb_pdb")
CIF_DIR = os.path.join(DATA, "rcsb_cif")
CIF2PDB = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/USalign-pylelab/cif2pdb"

# 找出失败占位文件（≤1200字节）
failed = []
for f in os.listdir(PDB_DIR):
    if os.path.getsize(os.path.join(PDB_DIR, f)) <= 1200:
        failed.append(f[:-4])

n_ok = 0
n_fail = 0
for pdb_id in failed:
    cif = os.path.join(CIF_DIR, f"{pdb_id}.cif")
    if not os.path.exists(cif) or os.path.getsize(cif) <= 5000:
        n_fail += 1
        continue
    out = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    r = subprocess.run([CIF2PDB, cif, out], capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 5000:
        n_ok += 1
    else:
        n_fail += 1
        if n_fail <= 5:
            print(f"转换失败 {pdb_id}: {r.stdout[-200:]} {r.stderr[-200:]}")

print(f"转换成功: {n_ok}  失败: {n_fail}")
