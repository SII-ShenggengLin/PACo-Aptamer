#!/usr/bin/env python3
"""把过滤后的 RNA 链切成独立 PDB 文件（qTMclust 输入）— AlignIF 复现第 3 步
- 输入: chains_pass.json（02 脚本输出，含 pdb/chain/length/resolution）
- 输出: data/chain_pdb/{pdb}_{chain}.pdb（统一链 ID 'A'，保留 TER）
"""
import json, os, shutil

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA, "rcsb_pdb")
OUT_DIR = os.path.join(DATA, "chain_pdb")
os.makedirs(OUT_DIR, exist_ok=True)

chains = json.load(open(os.path.join(DATA, "chains_pass.json")))
print(f"待切链: {len(chains)} 条")

n_ok = 0
for c in chains:
    src = os.path.join(PDB_DIR, f"{c['pdb']}.pdb")
    dst = os.path.join(OUT_DIR, f"{c['pdb']}_{c['chain']}.pdb")
    target_chain = c['chain']
    with open(src) as fin, open(dst, "w") as fout:
        in_target = False
        for line in fin:
            rec = line[:6].strip()
            if rec == "ATOM":
                chain = line[21]
                # 收集该链的原子；替换链 ID 为 A
                if chain == target_chain:
                    fout.write(line[:21] + "A" + line[22:])
                    in_target = True
                else:
                    if in_target:
                        fout.write("TER\n")
                        in_target = False
            elif rec == "TER":
                if in_target:
                    fout.write("TER\n")
                    in_target = False
            elif rec == "END":
                if in_target:
                    fout.write("TER\n")
                fout.write("END\n")
                break
            elif rec == "MODEL":
                pass  # 只取第一个模型
    n_ok += 1

print(f"切链完成: {n_ok}")

# 生成 qTMclust 的 chain list
with open(os.path.join(DATA, "chain_list.txt"), "w") as f:
    for c in chains:
        f.write(f"{c['pdb']}_{c['chain']}\n")
print("chain_list.txt 已生成")
