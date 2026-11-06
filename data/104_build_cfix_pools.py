#!/usr/bin/env python3
"""gRNAde 解码 C/G 互换修正：建 _cfix 修正池（2026-10-16）

bug 背景：17v2/26v2 推理脚本 seq_to_str 用 vocab=["A","C","G","U"]，
而 gRNAde 训练管线的 token 序是 A=0,G=1,C=2,U=3（src/constants.py
RNA_NUCLEOTIDES=["A","G","C","U"]）→ 写盘序列 C↔G 互换。
映射是确定性双射 → 对既有 .seqs 做 str.maketrans('CG','GC') 即得官方解码序列。
native 文件（真值序列，不经解码）原样复制。

输出（临时目录 + rename，绝不覆盖既有文件；目标已存在则报错退出）：
  data/survey/grnade_out_btest_v2_cfix/   （477 链 × 3 温度 + native）
  data/survey/grnade_out_av2_retrained_cfix/ （289 链 × 3 温度 + native）
"""
import os, shutil, sys

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
TR = str.maketrans("CG", "GC")


def fix_dir(src_name, dst_name):
    src = os.path.join(SUR, src_name)
    dst = os.path.join(SUR, dst_name)
    if os.path.exists(dst):
        raise RuntimeError(f"{dst} 已存在，拒绝覆盖")
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    n_seqs = n_native = 0
    for fn in sorted(os.listdir(src)):
        sp = os.path.join(src, fn)
        dp = os.path.join(tmp, fn)
        if fn.endswith(".seqs"):
            with open(sp) as f, open(dp + ".t", "w") as g:
                for line in f:
                    g.write(line.rstrip("\n").translate(TR) + "\n")
            os.replace(dp + ".t", dp)
            n_seqs += 1
        elif fn.endswith(".native"):
            shutil.copyfile(sp, dp)
            n_native += 1
        else:
            print(f"  [skip] 未知文件 {fn}")
    os.rename(tmp, dst)
    print(f"[cfix] {src_name} -> {dst_name}: {n_seqs} seqs + {n_native} native")


if __name__ == "__main__":
    fix_dir("grnade_out_btest_v2", "grnade_out_btest_v2_cfix")
    fix_dir("grnade_out_av2_retrained", "grnade_out_av2_retrained_cfix")
