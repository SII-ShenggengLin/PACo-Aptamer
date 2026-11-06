#!/usr/bin/env python3
"""第 55 步：B 层 test 权威性修复——坏链输入重建 + 重推理任务分发

背景（2026-08-16 审计）：
- RhoDesign B 层：11 条 s2 独有 test 链用了 43c 的坏文件（每残基仅 P 原子）→ 结构输入垃圾
- gRNAde B 层：3 条多拷贝链（1EI2_A/2KGP_A/6IZP_A）输入为 17/10 拷贝未去重 + 13 条
  bookkeeping 链 native 与 canonical 不一致（43b 丢残基/5'G 差异）
- PACo B 层：40 条链 native 与 canonical 不一致（3 条多拷贝拼接 425/250/290nt 严重错误）
- RhoDesign A 层：2KGP_A/6IZP_A 的 native 是 NMR 10 模型堆叠（250/290nt）

修复：一律以 canonical_v2（51 号）为唯一真值重建输入并重推理。
产物：/tmp/repair_*.json 任务文件 + survey/55_repair_manifest.json
"""
import os, json, shutil, subprocess

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
CANON = os.path.join(DATA_DIR, "canonical_v2")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"

canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
diffs = json.load(open("/tmp/btest_native_diffs.json"))

OLD11 = ["1EI2_A", "1O15_A", "1OLN_C", "2KGP_A", "6IZP_A", "6OD9_A", "6OD9_B",
         "6UET_A", "7AAV_Z", "7ABI_Z", "8G8Z_R"]

manifest = {}

# ---------- 1) RhoDesign B 层：fix4 目录换成 canonical 干净文件 ----------
fix4 = os.path.join(SUR, "rhodesign_btest_fix4")
rho_chains = sorted(set(OLD11) | set(diffs["RhoDesign"]))
for stem in rho_chains:
    shutil.copy(os.path.join(CANON, f"{stem}.pdb"), os.path.join(fix4, f"{stem}.pdb"))
    for f in os.listdir(os.path.join(SUR, "rhodesign_out_btest_v2")):
        if f.startswith(stem + "."):
            os.remove(os.path.join(SUR, "rhodesign_out_btest_v2", f))
json.dump([[c.rsplit("_", 1)[0], c.rsplit("_", 1)[1]] for c in rho_chains],
          open("/tmp/repair_rho_btest.json", "w"))
manifest["rhodesign_btest"] = rho_chains

# ---------- 2) gRNAde B 层：重建 16 条三件套 ----------
gdir = os.path.join(SUR, "grnade_inputs_btest_v2")
g_chains = diffs["gRNAde"]
for stem in g_chains:
    seq = canon[stem]["seq"]
    shutil.copy(os.path.join(CANON, f"{stem}.pdb"), os.path.join(gdir, f"{stem}.pdb"))
    with open(os.path.join(gdir, f"{stem}.seq"), "w") as f:
        f.write(seq + "\n")
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n", capture_output=True, text=True)
    ss = r.stdout.strip().split("\n")[-1].split()[0]
    with open(os.path.join(gdir, f"{stem}.ss"), "w") as f:
        f.write(ss + "\n")
    for f in os.listdir(os.path.join(SUR, "grnade_out_btest_v2")):
        if f.startswith(stem + "."):
            os.remove(os.path.join(SUR, "grnade_out_btest_v2", f))
json.dump([[c.rsplit("_", 1)[0], c.rsplit("_", 1)[1]] for c in g_chains],
          open("/tmp/repair_grnade_btest.json", "w"))
manifest["grnade_btest"] = g_chains

# ---------- 3) RhoDesign A 层：2 条 NMR 多模型链（canonical 为单模型单拷贝）----------
# 20v2 无 FIX 机制 → 用环境变量 RHOFIX_DIR（已加补丁）指向 canonical
a_chains = ["2KGP_A", "6IZP_A"]
for stem in a_chains:
    for f in os.listdir(os.path.join(SUR, "rhodesign_out_v2")):
        if f.startswith(stem + "."):
            os.remove(os.path.join(SUR, "rhodesign_out_v2", f))
json.dump([[c.rsplit("_", 1)[0], c.rsplit("_", 1)[1]] for c in a_chains],
          open("/tmp/repair_rho_alayer.json", "w"))
manifest["rhodesign_alayer"] = a_chains

# ---------- 4) PACo B 层：40 条（56 号脚本处理，此处仅记录）----------
manifest["paco_btest"] = diffs["PACo"]

json.dump(manifest, open(os.path.join(SUR, "55_repair_manifest.json"), "w"), indent=1)
print(json.dumps({k: len(v) for k, v in manifest.items()}, indent=1))
