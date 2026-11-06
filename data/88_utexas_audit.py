#!/usr/bin/env python3
"""第 88 步：UTexas Aptamer Database 扩样审计（16 号报告 §七.3 可选项）

目的：把"SELEX 适配体在模型似然下 off-manifold"的发现扩到第二个独立数据库
（UTexas aptamer DB，1,415 序列/489 论文，Zenodo 8387047 Sept2023 版）。

相位：
  --phase clean   解析 xlsx → RNA-only/ACGU/长度过滤/去重（与 79 号 AptaDB 序列合并去重，
                  已有结构的直接复用）→ 88_utexas_clean.json + fold fasta
  --phase leak    cd-hit-est 0.8 联聚类 vs 训练库（复用 79 号口径）
  --phase fold    RhoFold+ 新序列（GPU 1/2）
  --phase score   ens2 logP（self + randA 同 SS 随机对照）
  --phase report  与 AptaDB 合并的先验审计汇总 → 88_utexas_audit.json
"""
import os, sys, json, time, argparse, subprocess
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m77 = _imp("m77", os.path.join(ROOT, "data", "77_p0234_analysis.py"))
m79 = _imp("m79", os.path.join(ROOT, "data", "79_aptadb_retro.py"))

XLSX = "/tmp/utexas.xlsx"
FOLD_DIR = os.path.join(SUR, "88_rhofold")
CLEAN_J = os.path.join(SUR, "88_utexas_clean.json")
SCORED_J = os.path.join(SUR, "88_utexas_scored.json")
REPORT_J = os.path.join(SUR, "88_utexas_audit.json")
N_RANDOM = 10


def phase_clean():
    import pandas as pd
    df = pd.read_excel(XLSX)
    n0 = len(df)
    # 只取未修饰 ssRNA（2'-F/O-Me 等修饰化学模型未见，诚实剔除）
    sub = df[df["Type of Nucleic Acid"].astype(str).str.strip() == "ssRNA"]
    n_ss = len(sub)
    rows, seen = [], set()
    for _, r in sub.iterrows():
        s = str(r["Aptamer Sequence"]).upper().strip()
        if s.startswith("5'"):
            s = s[2:]
        if s.endswith("3'"):
            s = s[:-2]
        s = "".join(s.split())
        if not s or set(s) - set("ACGU"):
            continue
        if not (15 <= len(s) <= 300) or s in seen:
            continue
        seen.add(s)
        rows.append({"key": f"utx{len(rows)}", "seq": s,
                     "target": str(r.get("Target ", ""))[:80],
                     "kd_nM": float(r["Kd (nM)"]) if pd.notna(r.get("Kd (nM)")) else None})
    # 与 79 号 AptaDB 序列去重（结构可复用）
    apta = json.load(open(os.path.join(SUR, "79_aptadb_clean.json")))["entries"]
    apta_seqs = {e["seq"] for e in apta}
    rows = [r for r in rows if r["seq"] not in apta_seqs]
    json.dump({"n_raw": n0, "n_ssRNA": n_ss,
               "entries": rows}, open(CLEAN_J, "w"), indent=1)
    os.makedirs(FOLD_DIR, exist_ok=True)
    with open(os.path.join(FOLD_DIR, "utx_input.fa"), "w") as f:
        for r in rows:
            f.write(f">{r['key']}\n{r['seq']}\n")
    print(f"[88-clean] 原始 {n0} 行 → RNA/ACGU/去重/排 AptaDB 后 {len(rows)} 条")


def phase_leak():
    from subprocess import run
    import re
    d = json.load(open(CLEAN_J))
    entries = d["entries"]
    lib = json.load(open(os.path.join(ROOT, "data", "splits_v2", "seqs_all.json")))
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    stem2set = {s: k for k in ("train", "val", "test", "independent_test", "zero_shot_targets")
                for s in sp[k]}
    merged = os.path.join(SUR, "88_leak_in.fa")
    with open(merged, "w") as f:
        for e in entries:
            f.write(f">{e['key']}\n{e['seq']}\n")
        for stem, s in lib.items():
            if stem in stem2set:
                f.write(f">{stem}\n{s}\n")
    run(["cd-hit-est", "-i", merged, "-o", os.path.join(SUR, "88_leak_out.fa"),
         "-c", "0.8", "-n", "8", "-d", "0", "-T", "16", "-M", "0"],
        capture_output=True, text=True)
    clusters, cur = {}, None
    for line in open(os.path.join(SUR, "88_leak_out.fa.clstr")):
        if line.startswith(">Cluster"):
            cur = line.strip(); clusters[cur] = []
        else:
            m = re.search(r">(.*?)\.\.\.", line)
            if m and cur:
                clusters[cur].append(m.group(1))
    seq2cluster = {m: c for c, ms in clusters.items() for m in ms}
    n_train_cluster = 0
    for e in entries:
        c = seq2cluster.get(e["key"])
        e["leak"] = {"cluster_train": False}
        if c:
            sets = {stem2set[m] for m in clusters[c]
                    if not m.startswith("utx") and m in stem2set}
            if "train" in sets:
                e["leak"]["cluster_train"] = True
                n_train_cluster += 1
    d["leak_summary"] = {"n": len(entries), "n_cluster_train": n_train_cluster,
                         "n_clean": len(entries) - n_train_cluster}
    json.dump(d, open(CLEAN_J, "w"), indent=1)
    print(json.dumps(d["leak_summary"], indent=1))


def phase_fold():
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    d = json.load(open(CLEAN_J))
    items = [(e["seq"], e["key"]) for e in d["entries"]]

    def run_one(item, gpu):
        seq, key = item
        out_pdb = os.path.join(FOLD_DIR, key + ".pdb")
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 1000:
            return "cached"
        fa = os.path.join(FOLD_DIR, key + ".fa")
        with open(fa, "w") as f:
            f.write(f">{key}\n{seq}\n")
        wd = os.path.join(FOLD_DIR, "tmp_" + key)
        cmd = [m79.RHOFOLD_PY, "inference.py", "--input_fas", fa, "--output_dir", wd,
               "--device", f"cuda:{gpu}", "--single_seq_pred", "True", "--relax_steps", "0",
               "--ckpt", f"{m79.RHOFOLD_REPO}/pretrained/RhoFold_pretrained.pt"]
        try:
            r = subprocess.run(cmd, cwd=m79.RHOFOLD_REPO, capture_output=True, timeout=600)
            pred = os.path.join(wd, "unrelaxed_model.pdb")
            if r.returncode != 0 or not os.path.exists(pred):
                return "fail"
            os.rename(pred, out_pdb)
            shutil.rmtree(wd, ignore_errors=True)
            return "ok"
        except Exception:
            return "fail"

    t0 = time.time()
    stats = {"ok": 0, "cached": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_one, it, i % 2) for i, it in enumerate(items)]
        for i, f in enumerate(futs):
            stats[f.result()] = stats.get(f.result(), 0) + 1
            if (i + 1) % 50 == 0:
                print(f"[88-fold] {i+1}/{len(items)} {stats} | {time.time()-t0:.0f}s", flush=True)
    print(f"[88-fold] 完成 {stats} | {time.time()-t0:.0f}s")


def phase_score():
    import torch
    device = 0
    models, alpha = [], None
    for n in m79.ENS_CKPTS:
        m, a = m74_load(n, device)
        models.append(m); alpha = a
    alphabet = alpha
    from PACoAptamer.data_utils import ss_to_pairs
    bos = alphabet.get_idx("<cath>")

    @torch.no_grad()
    def logp(seqs, coords9, ss_pairs):
        import torch.nn.functional as F
        L = coords9.shape[0]
        coords = torch.tensor(coords9, dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(ss_pairs, dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        B = len(seqs)
        prev = torch.zeros(B, L, dtype=torch.long, device=device)
        tgt = torch.zeros(B, L, dtype=torch.long, device=device)
        for bi, sq in enumerate(seqs):
            toks = [alphabet.get_idx(c) for c in sq]
            prev[bi, 0] = bos
            prev[bi, 1:] = torch.tensor(toks[:-1], device=device)
            tgt[bi] = torch.tensor(toks, device=device)
        probs = None
        for m in models:
            eo = m.forward_embedding(coords, ss, lig, ligm, pmask, conf)
            eoB = {"encoder_out": [eo["encoder_out"][0].repeat(1, B, 1)],
                   "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(B, 1)],
                   "encoder_embedding": [], "encoder_states": []}
            logits, _ = m.decoder(prev, encoder_out=eoB)
            p = F.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
        probs = probs / len(models)
        return torch.log(probs + 1e-12).gather(1, tgt.unsqueeze(1)).squeeze(1).mean(dim=1).cpu().numpy()

    m74 = _imp("m74", os.path.join(ROOT, "data", "74_p00_iface_ablation.py"))
    d = json.load(open(CLEAN_J))
    scored = json.load(open(SCORED_J)) if os.path.exists(SCORED_J) else {}
    t0 = time.time()
    n_done = 0
    for e in d["entries"]:
        key = e["key"]
        if key in scored:
            continue
        pdb_p = os.path.join(FOLD_DIR, key + ".pdb")
        if not os.path.exists(pdb_p):
            continue
        coords9 = m79.coords9_from_pdb(pdb_p)
        if coords9 is None or coords9.shape[0] != len(e["seq"]):
            scored[key] = {"failed": "coords"}
            continue
        import zlib
        rng = np.random.RandomState(zlib.crc32(key.encode()) % (2**31))
        rand = ["".join(rng.choice(list("ACGU"), len(e["seq"]))) for _ in range(N_RANDOM)]
        structs = m77.fold_batch([e["seq"]] + rand)
        if not structs or len(structs) != N_RANDOM + 1:
            scored[key] = {"failed": "fold"}
            continue
        try:
            lp_self = float(logp([e["seq"]], coords9, ss_to_pairs(structs[0]))[0])
            lp_randA = logp(rand, coords9, ss_to_pairs(structs[0]))
        except RuntimeError:
            scored[key] = {"failed": "model"}
            continue
        scored[key] = {"logp_self": lp_self, "logp_self_per_nt": lp_self / len(e["seq"]),
                       "logp_randA": [float(x) for x in lp_randA], "len": len(e["seq"])}
        n_done += 1
        if n_done % 100 == 0:
            print(f"[88-score] {n_done} | {time.time()-t0:.0f}s", flush=True)
    json.dump(scored, open(SCORED_J, "w"))
    print(f"[88-score] ok {sum(1 for v in scored.values() if 'logp_self' in v)} / fail "
          f"{sum(1 for v in scored.values() if v.get('failed'))}")


def m74_load(n, device):
    m74 = _imp("m74", os.path.join(ROOT, "data", "74_p00_iface_ablation.py"))
    return m74.load_model(os.path.join(ROOT, "models", "PACoAptamer", n + ".pth"), device)


def phase_report():
    from scipy.stats import wilcoxon
    d = json.load(open(CLEAN_J))
    sc = json.load(open(SCORED_J))
    clean = []
    for e in d["entries"]:
        s = sc.get(e["key"])
        if s and "logp_self" in s and not e["leak"]["cluster_train"]:
            clean.append((e, s))
    dif = [s["logp_self"] - float(np.mean(s["logp_randA"])) for _, s in clean]
    out = {"db": "UTexas (Zenodo 8387047, Sept2023)", "n_clean": len(clean),
           "self_minus_randA": {"mean": float(np.mean(dif)),
                                "frac_negative": float(np.mean(np.array(dif) < 0))}}
    rng = np.random.RandomState(0)
    bs = [np.mean(rng.choice(dif, len(dif))) for _ in range(3000)]
    out["self_minus_randA"]["ci95"] = [float(np.percentile(bs, 2.5)),
                                       float(np.percentile(bs, 97.5))]
    # 合并 AptaDB（16 号同口径；exact-train ⊆ cluster_train，一并剔除与 79 号主分析一致）
    aptA_sc = json.load(open(os.path.join(SUR, "79_aptadb_scored.json")))["entries"]
    apta_clean = json.load(open(os.path.join(SUR, "79_aptadb_clean.json")))["entries"]
    ad = [aptA_sc[e["key"]]["logp_self"] - float(np.mean(aptA_sc[e["key"]]["logp_rand_sameSS"]))
          for e in apta_clean if e["key"] in aptA_sc and "logp_self" in aptA_sc[e["key"]]
          and not e["leak"]["cluster_train"] and e["leak"]["exact"] is None]
    merged = dif + ad
    out["merged_with_aptadb"] = {"n": len(merged), "mean": float(np.mean(merged)),
                                 "frac_negative": float(np.mean(np.array(merged) < 0))}
    json.dump(out, open(REPORT_J, "w"), indent=1)
    print(json.dumps(out, indent=1))



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['clean', 'leak', 'fold', 'score', 'report'])
    args = ap.parse_args()
    dict(clean=phase_clean, leak=phase_leak, fold=phase_fold,
         score=phase_score, report=phase_report)[args.phase]()
    if args.phase == 'report':
        pass


if __name__ == "__main__":
    main()
