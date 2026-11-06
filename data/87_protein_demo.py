#!/usr/bin/env python3
"""第 87 步：蛋白靶标可行性演示（蓝图 P2-12 / §八 8.1 主动防御"为何只做小分子"）

机制：PACo 的配体上下文输入是"任意原子集合 + 元素 + 相对 C1' 的 RBF"——蛋白口袋原子
天然可注入（零架构改动、零重训，蛋白上下文对模型完全 zero-shot）。
对照：五基线没有任何靶标上下文输入通道（定性差异）；定量看 PACo±蛋白上下文。

对象：B 层 test 477 中父 PDB 含蛋白链的 271 条里抽 30 条（种子固定）。
条件：none（空上下文，主表协议）vs protein（RNA 8Å 内蛋白口袋原子，≥120 原子时均匀下采样至 120）。
指标：蛋白界面残基（C1' 8Å 内）恢复率与全局恢复率，t=0.1 ×50 池内 best 与均值，
逐链配对 Wilcoxon。
输出：survey/87_protein_demo.json
"""
import os, sys, json, time
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "data"))
import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(ROOT, "data", "74_p00_iface_ablation.py"))
from PACoAptamer.data_utils import load_s1, lig_features
from scipy.spatial import cKDTree

AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET",
      "PHE","PRO","SER","THR","TRP","TYR","VAL"}
C1P_IDX = 1
N_STEMS = 30
N_SAMPLES = 50
CONTACT = 8.0
MAX_ATOMS = 120


def protein_atoms(pdb_id):
    """蛋白链 ATOM → (elems, coords)"""
    p = os.path.join(ROOT, "data", "rcsb_pdb", f"{pdb_id}.pdb")
    if not os.path.exists(p):
        return [], np.zeros((0, 3), dtype=np.float32)
    elems, coords = [], []
    for line in open(p):
        if not line.startswith("ATOM"):
            continue
        if line[17:20].strip() not in AA:
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        elems.append(line[76:78].strip() or "C")
        coords.append(xyz)
    return elems, np.array(coords, dtype=np.float32)


def main():
    import torch
    device = 0
    torch.manual_seed(42)
    models, alpha = [], None
    for n in ("s2_v2_seed43", "s2_v2_ssdo15"):
        m, a = m74.load_model(os.path.join(ROOT, "models", "PACoAptamer", n + ".pth"), device)
        models.append(m); alpha = a
    alphabet = alpha
    m39 = _imp("m39", os.path.join(ROOT, "data", "39_eval_paco_ensemble.py"))

    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp["test"]), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))
    # 含蛋白链的 test stem（长度 ≤300，控制推理时长）
    cand = []
    for stem in sorted(sp["test"]):
        if stem not in data or len(data[stem]["seq"]) > 300:
            continue
        pdb_id = stem.rsplit("_", 1)[0]
        elems, coords = protein_atoms(pdb_id)
        if len(elems) >= 50:
            cand.append(stem)
    rng = np.random.RandomState(42)
    stems = list(rng.choice(cand, min(N_STEMS, len(cand)), replace=False))
    print(f"[87] 候选 {len(cand)} → 抽样 {len(stems)}", flush=True)

    @torch.no_grad()
    def sample_ens_lig(d, lig_np, temperature, n_samples):
        L = len(d["seq"])
        coords = torch.tensor(d["coords9"], dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        if lig_np is not None and len(lig_np) > 0:
            lig = torch.tensor(lig_np, dtype=torch.float32, device=device).unsqueeze(0)
            ligm = torch.ones(1, lig_np.shape[0], dtype=torch.bool, device=device)
        else:
            lig = torch.zeros(1, 0, 21, device=device)
            ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        # 39 号集成采样逐模型概率平均（含 ligand 输入）
        import torch.nn.functional as F
        n = int(n_samples)
        K = len(models)
        enc_list = []
        for m in models:
            padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
            eo = m.forward_embedding(coords, ss, lig, ligm, padding_mask, conf)
            enc_list.append({"encoder_out": [eo["encoder_out"][0].repeat(1, n, 1)],
                             "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(n, 1)],
                             "encoder_embedding": [], "encoder_states": []})
        bos = alphabet.get_idx("<cath>")
        pred = torch.zeros(n, L + 1, dtype=torch.long, device=device)
        pred[:, 0] = bos
        states = [{} for _ in range(K)]
        for i in range(L):
            prev = pred[:, :i + 1]
            probs_sum = None
            for m, eo, st in zip(models, enc_list, states):
                logits, _ = m.decoder(prev, encoder_out=eo, incremental_state=st)
                probs = F.softmax(logits[:, :, -1] / temperature, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            pred[:, i + 1] = torch.multinomial(probs_sum / K, 1).squeeze(-1)
        return ["".join(alphabet.get_tok(t) for t in row.tolist()) for row in pred[:, 1:]]

    out = {}
    t0 = time.time()
    for i, stem in enumerate(stems):
        d = data[stem]
        native = d["seq"]
        c1p = np.nan_to_num(np.asarray(d["coords9"], dtype=np.float32)[:, C1P_IDX, :], nan=1e-5)
        pdb_id = stem.rsplit("_", 1)[0]
        elems, pcoords = protein_atoms(pdb_id)
        tree = cKDTree(pcoords)
        dist, _ = tree.query(c1p, k=1)          # 每残基到最近蛋白原子距离
        iface = dist < CONTACT
        if iface.sum() < 3:
            continue
        sel = dist < CONTACT + 4.0              # 口袋原子窗（12Å）
        sub = np.where(sel)[0]
        if len(sub) == 0:
            continue
        if len(sub) > MAX_ATOMS:
            sub = np.sort(rng.choice(sub, MAX_ATOMS, replace=False))
        lig_np = lig_features(pcoords[sub], [elems[j] for j in sub], c1p)
        rec = {}
        for cond, lf in (("none", None), ("protein", lig_np)):
            pool = sample_ens_lig(d, lf, 0.1, N_SAMPLES)
            recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
            ipos = [k for k, f in enumerate(iface) if f and native[k] != "N"]
            irecs = [sum(1 for k in ipos if s[k] == native[k]) / len(ipos) for s in pool]
            rec[cond] = {"overall_best": max(recs), "overall_mean": float(np.mean(recs)),
                         "iface_best": max(irecs), "iface_mean": float(np.mean(irecs))}
        out[stem] = {"n_iface": int(iface.sum()), "n_lig_atoms": int(len(sub)),
                     "len": len(native), **rec}
        print(f"[87] {i+1}/{len(stems)} {stem} iface={int(iface.sum())} atoms={len(sub)} | "
              f"iface_mean {rec['none']['iface_mean']:.3f}→{rec['protein']['iface_mean']:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)

    # 配对检验
    from scipy.stats import wilcoxon
    dif_i = [v["protein"]["iface_mean"] - v["none"]["iface_mean"] for v in out.values()]
    dif_o = [v["protein"]["overall_mean"] - v["none"]["overall_mean"] for v in out.values()]
    summary = {"n_chains": len(out),
               "iface_mean_none": float(np.mean([v["none"]["iface_mean"] for v in out.values()])),
               "iface_mean_protein": float(np.mean([v["protein"]["iface_mean"] for v in out.values()])),
               "overall_mean_none": float(np.mean([v["none"]["overall_mean"] for v in out.values()])),
               "overall_mean_protein": float(np.mean([v["protein"]["overall_mean"] for v in out.values()]))}
    for nm, dif in (("iface", dif_i), ("overall", dif_o)):
        if len(dif) >= 10:
            r = np.random.RandomState(0)
            bs = [np.mean(r.choice(dif, len(dif))) for _ in range(3000)]
            summary[f"delta_{nm}"] = {"mean": float(np.mean(dif)),
                                      "ci95": [float(np.percentile(bs, 2.5)),
                                               float(np.percentile(bs, 97.5))],
                                      "wilcoxon_p_two": float(wilcoxon(dif).pvalue)}
    json.dump({"summary": summary, "per_chain": out},
              open(os.path.join(SUR, "87_protein_demo.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
