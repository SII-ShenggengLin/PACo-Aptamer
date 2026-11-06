#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""111: 用官方 vocab(L2N={A:0,G:1,C:2,U:3}) 对 gRNAde cfix 池(grnade_out_btest_v2_cfix)
重打分,补齐 107 号置信度分析中 gRNAde 的 n/a 行。
打分逻辑逐字复刻 data/98_native_score.py run_grnade(同模型、同 featurize_custom、
同 L2N、同 t=0.1 池长度过滤与有效位掩码),仅输出池目录改为 cfix 版、输出文件改为 111 系。

用法:
  CUDA_VISIBLE_DEVICES=k <py> data/111_grnade_cfix_rescore.py --shard k --nshards 3
  <py> data/111_grnade_cfix_rescore.py --merge
产物: survey/111_part_grnade_cfix_shard{k}.json → survey/111_grnade_cfix_logp.json
"""
import os, sys, json, math, time, argparse

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")

TEST = sorted(json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))["test"])


def save_json(obj, fp):
    tmp = fp + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False)
    os.rename(tmp, fp)


def read_lines(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def run_shard(gpu, shard, nshards, limit=0):
    os.environ.setdefault("X3DNA", "/tmp/x3dna")
    os.environ.setdefault("ETERNAFOLD", "/tmp/eternafold")
    os.environ.setdefault("PROJECT_PATH", os.path.join(ROOT, "tools", "gRNAde"))
    sys.path.insert(0, os.path.join(ROOT, "tools", "gRNAde"))
    sys.path.insert(0, os.path.join(SUR, "pyshims"))
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from src.models import gRNAde
    from src.data.featurizer import RNAGraphFeaturizer

    IN_DIR = os.path.join(SUR, "grnade_inputs_btest_v2")
    OUT_DIR = os.path.join(SUR, "grnade_out_btest_v2_cfix")  # ← cfix 池(唯一改动)
    L2N = {"A": 0, "G": 1, "C": 2, "U": 3}  # 官方 featurizer 序

    model = gRNAde(node_in_dim=(15, 4), node_h_dim=(128, 16),
                   edge_in_dim=(132, 3), edge_h_dim=(64, 4),
                   num_layers=4, drop_rate=0.5, out_dim=4)
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "gRNAde_btrain_v2", "best_checkpoint.h5"),
                                     map_location="cpu"))
    model = model.to(f"cuda:{gpu}").eval()
    featurizer = RNAGraphFeaturizer(split="test", radius=0.0, top_k=32, num_rbf=32,
                                    num_posenc=32, max_num_conformers=1,
                                    noise_scale=0.1, drop_prob_3d=0.75)

    def featurize_custom(stem):
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
        return featurizer.featurize(rna)

    def mean_logp(batch, tok, valid):
        batch.seq = tok
        with torch.no_grad():
            logits = model(batch)
        lp = F.log_softmax(logits, dim=-1)
        per = lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return per[valid].mean().item()

    part_fp = os.path.join(SUR, f"111_part_grnade_cfix_shard{shard}.json")
    part = json.load(open(part_fp)) if os.path.exists(part_fp) else {}
    my = [s for i, s in enumerate(TEST) if i % nshards == shard]
    if limit:
        my = my[:limit]
    t0 = time.time()
    n_done = 0
    for stem in my:
        if stem in part and "logp_native" in part[stem]:
            continue
        try:
            batch = featurize_custom(stem).to(f"cuda:{gpu}")
            native = read_lines(os.path.join(OUT_DIR, f"{stem}.native"))[0]
            L = len(native)
            valid = torch.tensor([c in L2N for c in native], device=f"cuda:{gpu}")
            tok_nat = torch.tensor([L2N.get(c, 0) for c in native], dtype=torch.long,
                                   device=f"cuda:{gpu}")
            lp_nat = mean_logp(batch, tok_nat, valid)
            pool = [s for s in read_lines(os.path.join(OUT_DIR, f"{stem}.0.1.seqs"))
                    if len(s) == L]
            sample_logp = []
            BS = 50
            for i in range(0, len(pool), BS):
                chunk = pool[i:i + BS]
                graphs = []
                for s in chunk:
                    g = batch.clone()
                    g.seq = torch.tensor([L2N[c] for c in s], dtype=torch.long,
                                         device=f"cuda:{gpu}")
                    graphs.append(g)
                bb = Batch.from_data_list(graphs)
                with torch.no_grad():
                    logits = model(bb)
                lp = F.log_softmax(logits, dim=-1).view(len(chunk), L, 4)
                tok_b = torch.stack([g.seq for g in graphs])
                per = lp.gather(-1, tok_b.unsqueeze(-1)).squeeze(-1)
                sample_logp.extend(per[:, valid].mean(-1).tolist())
            part[stem] = {"logp_native": lp_nat, "ppl_native": math.exp(-lp_nat),
                          "n_pos": int(valid.sum()), "len": L,
                          "sample_logp_t01": sample_logp}
            n_done += 1
            if n_done % 25 == 0:
                save_json(part, part_fp)
                print(f"  [shard{shard}] {n_done}/{len(my)} ppl~{math.exp(-lp_nat):.2f}",
                      flush=True)
        except Exception as e:
            part[stem] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {stem}: {e}", flush=True)
    save_json(part, part_fp)
    ok = sum(1 for v in part.values() if "logp_native" in v)
    print(f"[shard{shard}] 完成 {ok}/{len(my)} ({(time.time()-t0)/60:.1f} min)", flush=True)


def merge():
    out = {}
    for k in range(3):
        fp = os.path.join(SUR, f"111_part_grnade_cfix_shard{k}.json")
        out.update(json.load(open(fp)))
    bad = {s: v for s, v in out.items() if "logp_native" not in v}
    assert not bad, f"未成功链: {list(bad)[:5]}"
    assert set(out) == set(TEST), f"链数不符 {len(out)} vs {len(TEST)}"
    # 校验 1: native logp 须与 98 号(同一 native/输入/模型/官方 vocab)逐链一致
    p98 = json.load(open(os.path.join(SUR, "98_part_grnade.json")))
    diffs = [abs(out[s]["logp_native"] - p98[s]["logp_native"]) for s in TEST
             if "logp_native" in p98.get(s, {})]
    max_d = max(diffs)
    print(f"[merge] native logp vs 98: max|Δ| = {max_d:.3e} (n={len(diffs)})")
    assert max_d < 1e-4, "native logp 与 98 不一致,打分路径有误"
    # 校验 2: cfix 样本 logp 平均应高于 98 存档(对 C/G 互换错序列)的同池均值
    dlt = []
    for s in TEST:
        old = p98[s].get("sample_logp_t01")
        new = out[s]["sample_logp_t01"]
        if old and new and len(old) == len(new):
            dlt.append(sum(new) / len(new) - sum(old) / len(old))
    import statistics
    print(f"[merge] cfix−swapped 样本 logp 差: mean={statistics.mean(dlt):+.4f} nat/nt, "
          f">0 链占比={sum(d > 0 for d in dlt)}/{len(dlt)}")
    meta = {
        "step": "111_grnade_cfix_rescore",
        "date": "2026-10-19",
        "purpose": "官方 vocab 重打分 gRNAde cfix 池,补 107 置信度 gRNAde n/a 行",
        "protocol": "逐字复刻 98_native_score.py run_grnade(官方 L2N={A:0,G:1,C:2,U:3}),"
                    "样本池改为 survey/grnade_out_btest_v2_cfix 的 t=0.1 池(长度过滤后逐样本"
                    " teacher-forced mean logP/nt,有效位掩码=native ACGU 位)",
        "model": "models/gRNAde_btrain_v2/best_checkpoint.h5",
        "validation": {"native_logp_max_abs_diff_vs_98": max_d,
                       "sample_logp_delta_vs_swapped_mean": statistics.mean(dlt)},
    }
    save_json({"meta": meta, "per_chain": out},
              os.path.join(SUR, "111_grnade_cfix_logp.json"))
    print(f"[merge] 落盘 111_grnade_cfix_logp.json ({len(out)} 链)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_shard(args.gpu, args.shard, args.nshards, args.limit)
