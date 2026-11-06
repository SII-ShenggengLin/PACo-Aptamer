#!/usr/bin/env python3
"""第 43 步：为 A 层重评估构建 v2 推理输入（新划分）

A 层 = indep 235 ∪ zero_shot 93 = 289 链（splits_v2.json）
- 输入：rcsb_pdb 切链（16 号逻辑，链重命名 A，丢弃修饰残基）+ RNAfold SS
- 69 链复用旧 grnade_inputs（口径一致：同为 rcsb 切链 + RNAfold）
- 兜底：chain_pdb / chain_pdb_v2（BGSU 单链文件，含修饰残基——用 manifest 序列做 .seq）
输出：survey/grnade_inputs_v2/{stem}.pdb + .seq + .ss
"""
import os, re, json, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA, "rcsb_pdb")
OUT_DIR = os.path.join(DATA, "survey", "grnade_inputs_v2")
OLD_DIR = os.path.join(DATA, "survey", "grnade_inputs")
CP_DIR = os.path.join(DATA, "chain_pdb")
CP2_DIR = os.path.join(DATA, "chain_pdb_v2")
os.makedirs(OUT_DIR, exist_ok=True)
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
STD = {"A", "C", "G", "U"}

sp = json.load(open(os.path.join(DATA, "splits_v2", "splits_v2.json")))
A_CHAINS = sorted(set(sp["independent_test"]) | set(sp["zero_shot_targets"]))
manifest = {e["pdb"]: e for e in json.load(open(os.path.join(DATA, "survey", "eval_manifest_v2.json")))}
print(f"A 层链: {len(A_CHAINS)}", flush=True)

def split_chain(pdb_id, chain, out_stem):
    """从条目 PDB 切出单链（ATOM + 该链 HETATM 修饰），链重命名 A（16 号逻辑）"""
    src = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    lines_out = []
    seq = []
    seen = set()
    with open(src) as f:
        for line in f:
            rec = line[:6].strip()
            if len(line) < 27:
                continue
            if rec in ("ATOM", "HETATM") and line[21] == chain and line[16] in (" ", "A"):
                rn = line[17:20].strip()
                if rn not in STD:
                    continue
                new = line[:21] + "A" + line[22:]
                lines_out.append(new)
                if rec == "ATOM":
                    key = (rn, line[22:26].strip(), line[26].strip())
                    if key not in seen:
                        seen.add(key)
                        seq.append(rn)
            elif rec == "TER" and len(line) >= 22 and line[21] == chain:
                lines_out.append(line)
            elif rec == "END":
                lines_out.append(line)
    if not lines_out or not seq:
        return None
    with open(out_stem + ".pdb", "w") as f:
        f.writelines(lines_out)
    return "".join(seq)

def predict_ss(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None

def build_one(stem):
    pdb_id, chain = stem.split("_")
    # 已存在且完整 → 跳过
    if all(os.path.exists(os.path.join(OUT_DIR, stem + x)) and
           os.path.getsize(os.path.join(OUT_DIR, stem + x)) > 0
           for x in (".pdb", ".seq", ".ss")):
        return (stem, "ok_exist")
    # 1) 复用旧 grnade_inputs（口径一致）
    if all(os.path.exists(os.path.join(OLD_DIR, stem + x))
           for x in (".pdb", ".seq", ".ss")):
        for x in (".pdb", ".seq", ".ss"):
            os.system(f"cp {os.path.join(OLD_DIR, stem + x)} {os.path.join(OUT_DIR, stem + x)}")
        return (stem, "reuse_old")
    # 2) rcsb_pdb 切链 + RNAfold
    seq = split_chain(pdb_id, chain, os.path.join(OUT_DIR, stem))
    if seq:
        with open(os.path.join(OUT_DIR, stem + ".seq"), "w") as f:
            f.write(seq + "\n")
        ss = predict_ss(seq)
        if ss:
            with open(os.path.join(OUT_DIR, stem + ".ss"), "w") as f:
                f.write(ss + "\n")
            return (stem, "rcsb_cut")
        return (stem, "no_ss")
    # 3) 兜底：chain_pdb / chain_pdb_v2（BGSU 单链，链已重命名 A，可能含修饰）
    for d in (CP_DIR, CP2_DIR):
        p = os.path.join(d, stem + ".pdb")
        if os.path.exists(p):
            os.system(f"cp {p} {os.path.join(OUT_DIR, stem + '.pdb')}")
            # 序列优先用 manifest（含修饰映射，与 PDB 残基数一致），否则从 PDB 提取
            entry = manifest.get(pdb_id, {})
            mseq = entry.get("sequences", {}).get(chain) if entry else None
            if mseq:
                with open(os.path.join(OUT_DIR, stem + ".seq"), "w") as f:
                    f.write(mseq + "\n")
            else:
                return (stem, "cp_no_seq")
            ss = predict_ss(mseq)
            if ss:
                with open(os.path.join(OUT_DIR, stem + ".ss"), "w") as f:
                    f.write(ss + "\n")
                return (stem, "chainpdb_fb")
            return (stem, "cp_no_ss")
    return (stem, "no_pdb")

if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build_one, s): s for s in A_CHAINS}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((futs[fut], f"EXC:{e}"))
    from collections import Counter
    c = Counter(r[1] for r in results)
    print("结果统计:", dict(c), flush=True)
    for r in results:
        if r[1] != "ok_exist" and not r[1].startswith("reuse") and r[1] != "rcsb_cut":
            print("  特殊:", r, flush=True)
    n_ok = sum(1 for r in results if all(
        os.path.exists(os.path.join(OUT_DIR, r[0] + x)) and
        os.path.getsize(os.path.join(OUT_DIR, r[0] + x)) > 0
        for x in (".pdb", ".seq", ".ss")))
    print(f"完成: {n_ok}/{len(A_CHAINS)} 链三件套齐备", flush=True)
