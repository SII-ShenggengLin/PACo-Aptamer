#!/usr/bin/env python3
"""第 103 步：gRNAde 在 RhoFold+ 预测骨架上的批量推理（86_sctm 193 链）

- 输入三件套 survey/grnade_inputs_predbb/{stem}.pdb/.seq/.ss（43b 口径）：
  .pdb = 86_sctm/native_{stem}.pdb 清洗（ATOM-only，链 ID 0→A）；
  .seq = canonical_v2 native 序列；.ss = RNAfold --noSS 预测（对 native）
- 推理：26v2 逻辑（featurize_custom 手工解析 P/C4'/N1|N9 + 外部 .ss）；
  ckpt models/gRNAde_btrain_v2/best_checkpoint.h5；t=0.1/0.5/1.0；
  ≤200nt 采 100、≤400nt 采 50、更长采 30
- 【修正】token→字母解码用官方序 ["A","G","C","U"]（featurizer letter_to_num 序）。
  26v2 的 ["A","C","G","U"] 与训练 token 序 C/G 互换（98 步已实证：训练管线
  dataset.py→RNAGraphFeaturizer 用 {A:0,G:1,C:2,U:3}；teacher-forced native logP
  官方序 5/5 链显著更高；40 链×100 样本配对对比官方解码 +0.02 恢复率）。
  晶体骨架对照请用 103 步指标脚本里的 --recalc-crystal（对既有 .seqs 做 C↔G 互换重算）。
- 指标：63 号口径（best=三温度全池内最大同一性；t01 mean=t=0.1 池逐样本平均；
  native=canonical_v2；'N' 位跳过——193 链无 N）
- 效率探针：前 6 链墙钟 + torch.cuda.max_memory_allocated → survey/99_efficiency_gRNAde.json
输出：survey/grnade_out_predbb/{stem}.{temp}.seqs + {stem}.native
      survey/103_grnade_predbb.json（tmp→rename；断点续跑）
用法：CUDA_VISIBLE_DEVICES=2 <py> data/103_run_grnade_predbb.py [--workers 1]
"""
import os, sys, json, subprocess, argparse, time
import numpy as np

os.environ.setdefault("X3DNA", "/tmp/x3dna")
os.environ.setdefault("ETERNAFOLD", "/tmp/eternafold")
os.environ.setdefault("PROJECT_PATH", "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
SRC_BB = os.path.join(SUR, "86_sctm")
IN_DIR = os.path.join(SUR, "grnade_inputs_predbb")
OUT_DIR = os.path.join(SUR, "grnade_out_predbb")
CKPT = os.path.join(ROOT, "models", "gRNAde_btrain_v2", "best_checkpoint.h5")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
TEMPS = [0.1, 0.5, 1.0]
N_SAMPLES = 100

sys.path.insert(0, os.path.join(ROOT, "tools", "gRNAde"))
sys.path.insert(0, os.path.join(SUR, "pyshims"))  # cpdb stub（featurize_custom 路径不调用）


def load_chains():
    d = json.load(open(os.path.join(SUR, "86_sctm.json")))
    return sorted(d["per_chain"].keys())


def samples_for_len(n):
    if n <= 200:
        return N_SAMPLES
    if n <= 400:
        return 50
    return 30


# ---------- 输入准备（43b 口径） ----------

def clean_pdb(stem):
    """86_sctm/native_{stem}.pdb → ATOM-only + 链 ID 改 A"""
    src = os.path.join(SRC_BB, f"native_{stem}.pdb")
    lines_out = []
    with open(src) as f:
        for line in f:
            if line.startswith("ATOM"):
                lines_out.append(line[:21] + "A" + line[22:])
            elif line.startswith(("TER", "END")):
                lines_out.append(line)
    with open(os.path.join(IN_DIR, stem + ".pdb"), "w") as f:
        f.writelines(lines_out)


def predict_ss(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    import re
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None


def prep_inputs(chains):
    os.makedirs(IN_DIR, exist_ok=True)
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    n_new, n_bad = 0, 0
    for stem in chains:
        outs = [os.path.join(IN_DIR, stem + x) for x in (".pdb", ".seq", ".ss")]
        if all(os.path.exists(o) and os.path.getsize(o) > 0 for o in outs):
            continue
        seq = canon[stem]["seq"]
        clean_pdb(stem)
        with open(os.path.join(IN_DIR, stem + ".seq"), "w") as f:
            f.write(seq + "\n")
        ss = predict_ss(seq.replace("N", "A"))  # RNAfold 不认 N，占位后该位结构无意义（gRNAde 侧同位剔除）
        if ss is None:
            n_bad += 1
            print(f"  [prep] {stem} RNAfold 失败", flush=True)
            continue
        with open(os.path.join(IN_DIR, stem + ".ss"), "w") as f:
            f.write(ss + "\n")
        n_new += 1
    print(f"[prep] 三件套 {len(chains)} 链（新建 {n_new}，RNAfold 失败 {n_bad}）", flush=True)


# ---------- 推理 ----------

def featurize_custom(featurizer, stem, torch):
    """26v2 逻辑：手工解析 P/C4'/N1|N9 三原子 + 外部 .ss"""
    pdb_path = os.path.join(IN_DIR, stem + ".pdb")
    seq = open(os.path.join(IN_DIR, stem + ".seq")).read().strip()
    ss = open(os.path.join(IN_DIR, stem + ".ss")).read().strip()
    seqf = seq.replace("N", "A") if "N" in seq else seq
    residues, order = {}, []
    for line in open(pdb_path):
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
    if len(order) != len(seqf):
        raise ValueError(f"{stem}: PDB 残基数 {len(order)} != 序列 {len(seqf)}")
    coords = []
    for i, key in enumerate(order):
        rn = seqf[i]
        base_atom = "N1" if rn in "CU" else "N9"
        pts = [residues[key].get(a, (1e-5, 1e-5, 1e-5)) for a in ["P", "C4'", base_atom]]
        coords.append(pts)
    rna = {"sequence": seqf, "coords_list": [torch.tensor(coords, dtype=torch.float32)],
           "sec_struct_list": [ss]}
    return featurizer.featurize(rna), seq


def run_inference(chains, gpu, probe_n=6):
    import torch
    from src.models import gRNAde
    from src.data.featurizer import RNAGraphFeaturizer

    model = gRNAde(node_in_dim=(15, 4), node_h_dim=(128, 16),
                   edge_in_dim=(132, 3), edge_h_dim=(64, 4),
                   num_layers=4, drop_rate=0.5, out_dim=4)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.to(f"cuda:{gpu}").eval()
    featurizer = RNAGraphFeaturizer(split="test", radius=0.0, top_k=32, num_rbf=32,
                                    num_posenc=32, max_num_conformers=1,
                                    noise_scale=0.1, drop_prob_3d=0.75)
    VOCAB = ["A", "G", "C", "U"]  # 官方 featurizer 序（见模块 docstring）
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {"ok": 0, "fail": 0, "cached": 0}
    probe_recs = []
    n_seen = 0
    for stem in chains:
        outs = [os.path.join(OUT_DIR, f"{stem}.{t}.seqs") for t in TEMPS]
        if all(os.path.exists(o) and os.path.getsize(o) > 0 for o in outs):
            stats["cached"] += 1
            continue
        do_probe = n_seen < probe_n
        n_seen += 1
        try:
            if do_probe:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            batch, native = featurize_custom(featurizer, stem, torch)
            batch = batch.to(f"cuda:{gpu}")
        except Exception as e:
            stats["fail"] += 1
            print(f"  [FAIL] {stem} featurize: {e}", flush=True)
            continue
        with open(os.path.join(OUT_DIR, f"{stem}.native"), "w") as f:
            f.write(native + "\n")
        n_samp = samples_for_len(len(native))
        ok_all = True
        for temp, out_f in zip(TEMPS, outs):
            try:
                with torch.no_grad():
                    seqs = model.sample(batch, n_samples=n_samp, temperature=temp)
                tmp_f = out_f + ".tmp"
                with open(tmp_f, "w") as f:
                    for s in seqs:
                        f.write("".join(VOCAB[i] for i in s.tolist()) + "\n")
                os.rename(tmp_f, out_f)
            except Exception as e:
                ok_all = False
                print(f"  [FAIL] {stem} temp={temp}: {e}", flush=True)
        if ok_all:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
        if do_probe:
            probe_recs.append({"stem": stem, "len": len(native), "n_samples_per_temp": n_samp,
                               "wall_s": round(time.time() - t0, 3),
                               "gpu_mem_peak_mib": round(torch.cuda.max_memory_allocated() / 1e6, 1),
                               "ok": ok_all})
            print(f"  [probe] {stem}: wall={probe_recs[-1]['wall_s']}s mem={probe_recs[-1]['gpu_mem_peak_mib']}MiB", flush=True)
        if stats["ok"] % 20 == 0:
            print(f"  进度 ok={stats['ok']} fail={stats['fail']} cached={stats['cached']}", flush=True)
    if probe_recs:
        fp = os.path.join(SUR, "99_efficiency_gRNAde.json")
        tmp = fp + ".tmp"
        json.dump({"method": "gRNAde", "setting": "predbb t=0.1/0.5/1.0 三温度采样（每温度 n=按链长）",
                   "note": "墙钟=featurize+三温度采样全程；显存=torch.cuda.max_memory_allocated",
                   "records": probe_recs}, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.rename(tmp, fp)
        print(f"[probe] 写入 {fp}", flush=True)
    print(f"[gRNAde-predbb] 完成: {stats}", flush=True)


# ---------- 指标（63 号口径） ----------

def rec_chain(pool, native):
    npos = [i for i, c in enumerate(native) if c != "N"]
    alls = [s for t in TEMPS for s in pool[t] if len(s) == len(native)]
    t01 = [s for s in pool[0.1] if len(s) == len(native)]
    if not alls or not t01 or not npos:
        return None
    rec_best = max(sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in alls)
    rec_t01 = float(np.mean([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in t01]))
    return rec_best, rec_t01, len(alls)


def compute_metrics(chains):
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    results, failed = {}, []
    for stem in chains:
        native = canon[stem]["seq"]
        pool = {}
        for t in TEMPS:
            fp = os.path.join(OUT_DIR, f"{stem}.{t}.seqs")
            pool[t] = [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []
        r = rec_chain(pool, native)
        if r is None:
            failed.append(stem)
            continue
        results[stem] = {"len": len(native), "recovery_best": r[0],
                         "recovery_t01_mean": r[1], "n_pool": r[2]}
    summary = {
        "n_chains": len(results), "coverage": f"{len(results)}/{len(chains)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])) if results else None,
        "note": "RhoFold+ 预测骨架(86_sctm) × gRNAde；官方 token 序解码（修正 26v2 C/G 互换）；口径同 btest_v2_main_table",
        "failed_chains": failed,
    }
    fp = os.path.join(SUR, "103_grnade_predbb.json")
    tmp = fp + ".tmp"
    json.dump({"summary": summary, "per_chain": results}, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.rename(tmp, fp)
    print(f"[指标] 写入 {fp}: best_mean={summary['recovery_best_mean']}, t01_mean={summary['recovery_t01_mean_mean']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA_VISIBLE_DEVICES 限制后的逻辑设备号")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()
    chains = load_chains()
    if not args.metrics_only:
        prep_inputs(chains)
    if not args.skip_inference and not args.metrics_only:
        run_inference(chains, args.gpu)
    compute_metrics(chains)
