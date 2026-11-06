#!/usr/bin/env python3
"""第 24 步：为 ODesign 生成 components.v20240608.cif.rdkit_mol.pkl

ODesign 期望 {ccd_code: rdkit.Mol}，mol 需带：
- atom_map: dict[str, int]  原子名 → 原子索引
- ref_conf_id: int          参考构象 id（用 CCD 坐标的 conformer）
- ref_mask: np.ndarray      原子掩码（全 True）
用 PDBeEurope/ccdutils（pdbeccdutils.read_pdb_components_file）解析全量 CCD。
"""
import os, pickle, logging
import numpy as np
import rdkit.Chem as Chem
from pdbeccdutils.core import ccd_reader

logging.basicConfig(level=logging.ERROR)

CCD_CIF = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/ODesign/data/components.v20240608.cif"
OUT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/ODesign/data/components.v20240608.cif.rdkit_mol.pkl"

def main():
    print("解析 components.cif（~50K 条目，需数分钟）...", flush=True)
    results = ccd_reader.read_pdb_components_file(CCD_CIF, sanitize=True)
    print(f"解析完成: {len(results)} 条目", flush=True)

    out = {}
    n_fail = 0
    for i, (code, res) in enumerate(results.items()):
        try:
            mol = res.component.mol
            if mol is None or mol.GetNumAtoms() == 0:
                n_fail += 1
                continue
            mol = Chem.Mol(mol)  # 深拷贝，避免共享变异
            # 原子名 → 索引
            atom_map = {}
            for atom in mol.GetAtoms():
                try:
                    nm = atom.GetProp("name")
                except KeyError:
                    nm = atom.GetSymbol()
                atom_map[nm] = atom.GetIdx()
            # 参考构象
            if mol.GetNumConformers() == 0:
                n_fail += 1
                continue
            mol.ref_conf_id = 0
            mol.atom_map = atom_map
            mol.ref_mask = np.ones(mol.GetNumAtoms(), dtype=bool)
            out[code] = mol
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  [{code}] fail: {e}", flush=True)
        if (i + 1) % 5000 == 0:
            print(f"进度: {i+1}/{len(results)} 有效 {len(out)}", flush=True)

    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"完成: {len(out)} 条目（失败 {n_fail}）→ {OUT}", flush=True)

if __name__ == "__main__":
    main()
