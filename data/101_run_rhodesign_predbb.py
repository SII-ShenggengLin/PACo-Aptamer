#!/usr/bin/env python3
"""第 101 步：RhoDesign 在 RhoFold+ 预测骨架上的批量逆折叠推理（预测骨架口径）

- 输入：survey/101_predbb_inputs/{stem}.pdb（86_sctm/native_{stem}.pdb 的符号链接，
  193 条 test 链的 RhoFold+ 预测结构；输出去 native_ 前缀）
- 模型：models/RhoDesign_btrain_v2/best_val.pth（与晶体 btest 同权重，变量只有骨架来源）
- 采样：与 29v2 btest 同口径：temperature 0.1/0.5/1.0；≤200nt ×100，≤400nt ×50，更长 ×30
- 指标：池内 best / t=0.1 均值（21btest 同口径），native 序列取自
  survey/rhodesign_out_btest_v2/{stem}.native（晶体口径 manifest）
- 输出：survey/rhodesign_out_predbb/{stem}.{temp}.seqs + survey/101_rhodesign_predbb.json
  （逐链明细 + 汇总；每完成 30 链落盘一次，重启自动跳过已完成链）
- 效率探针：长度分位取 6 条代表链（最短/p20/p40/p60/p80/最长）优先跑，记录每链墙钟与
  torch.cuda.max_memory_allocated → survey/99_efficiency_RhoDesign.json
- 环境：rhodesign_service_env（torch 1.12）；GPU：CUDA_VISIBLE_DEVICES=1 ... --gpu 0

用法：CUDA_VISIBLE_DEVICES=1 python data/101_run_rhodesign_predbb.py --gpu 0
"""
import os, sys, json, argparse, random, time
import numpy as np
import torch

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoDesign/src")
from RhoDesign_without2d import RhoDesignModel   # no_ss 版本
from alphabet import Alphabet
from util import load_structure, extract_coords_from_structure

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
SUR = os.path.join(DATA_DIR, "survey")
IN_DIR = os.path.join(SUR, "101_predbb_inputs")
NAT_DIR = os.path.join(SUR, "rhodesign_out_btest_v2")
OUT_DIR = os.path.join(SUR, "rhodesign_out_predbb")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(SUR, "101_rhodesign_predbb.json")
EFF_JSON = os.path.join(SUR, "99_efficiency_RhoDesign.json")
CKPT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/RhoDesign_btrain_v2/best_val.pth"
TEMPS = [0.1, 0.5, 1.0]
N_SAMPLES = 100
CHECKPOINT_EVERY = 30


class args_class:
    def __init__(self):
        self.local_rank = -1
        self.device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        self.epochs = 100; self.lr = 1e-5; self.batch_size = 1
        self.encoder_embed_dim = 512; self.decoder_embed_dim = 512; self.dropout = 0.1
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256; self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32; self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3; self.gvp_dropout = 0.1
        self.encoder_layers = 3; self.encoder_attention_heads = 4
        self.attention_dropout = 0.1; self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3; self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512


def samples_for_len(n):
    if n <= 200: return N_SAMPLES
    if n <= 400: return 50
    return 30


def chain_done(stem):
    return all(os.path.exists(os.path.join(OUT_DIR, f"{stem}.{t}.seqs")) for t in TEMPS)


def metrics_for(stem):
    """读输出池算 best/t01（21btest 口径）"""
    native = open(os.path.join(NAT_DIR, f"{stem}.native")).read().strip()
    pools = {}
    for t in TEMPS:
        pools[t] = [l.strip() for l in open(os.path.join(OUT_DIR, f"{stem}.{t}.seqs")) if l.strip()]
    def rec(s):
        return sum(1 for a, b in zip(s, native) if a == b) / len(native)
    all_recs = [rec(s) for t in TEMPS for s in pools[t]]
    t01 = [rec(s) for s in pools[0.1]]
    return {"len": len(native), "recovery_best": max(all_recs),
            "recovery_t01_mean": float(np.mean(t01)),
            "n_pool": len(all_recs), "n_t01": len(t01)}


def write_metrics_json(stems, note_extra=""):
    results, failed = {}, {}
    for s in stems:
        if chain_done(s):
            try:
                results[s] = metrics_for(s)
            except Exception as e:
                failed[s] = str(e)
        else:
            failed[s] = "incomplete"
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(stems)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])) if results else None,
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])) if results else None,
        "note": ("RhoDesign btrain_v2 权重 × RhoFold+ 预测骨架（101_predbb_inputs，193 test 链）；"
                 "best=3 温度池内 max，t01=temp0.1 池均值（21btest 口径）；" + note_extra),
    }
    json.dump({"summary": summary, "per_chain": results, "failed": failed},
              open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--taskfile", type=str, default=None)
    args = ap.parse_args()
    random.seed(1)

    stems = json.load(open(args.taskfile)) if args.taskfile else \
        sorted(f[:-4] for f in os.listdir(IN_DIR) if f.endswith(".pdb"))
    # native 长度（决定采样数与探针选链）
    nat_len = {s: len(open(os.path.join(NAT_DIR, s + ".native")).read().strip()) for s in stems}

    # 效率探针：长度分位 6 链优先
    by_len = sorted(stems, key=lambda s: nat_len[s])
    q = [0, len(by_len) // 5, 2 * len(by_len) // 5, 3 * len(by_len) // 5,
         4 * len(by_len) // 5, len(by_len) - 1]
    probe = sorted({by_len[i] for i in q}, key=lambda s: nat_len[s])
    todo = probe + [s for s in stems if s not in probe]
    print(f"总 {len(stems)} 链 | 探针 {[(s, nat_len[s]) for s in probe]}", flush=True)

    args_cls = args_class()
    dictionary = Alphabet(["A", "G", "C", "U", "X"])
    model = RhoDesignModel(args_cls, dictionary)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.cuda(args.gpu)
    model.eval()

    probe_rec = {}
    stats = {"ok": 0, "fail": 0, "skip": 0}
    n_new = 0
    for i, stem in enumerate(todo):
        if chain_done(stem):
            stats["skip"] += 1
            continue
        pdb_path = os.path.join(IN_DIR, stem + ".pdb")
        try:
            structure = load_structure(pdb_path)
            coords, _native = extract_coords_from_structure(structure)
        except Exception as e:
            stats["fail"] += 1
            print(f"  [FAIL] {stem} load: {e}", flush=True)
            continue
        native = open(os.path.join(NAT_DIR, stem + ".native")).read().strip()
        n_samp = samples_for_len(len(native))
        is_probe = stem in probe and stem not in probe_rec
        if is_probe:
            torch.cuda.reset_peak_memory_stats(args.gpu)
            t0 = time.time()
        per_temp_t = {}
        ok = True
        for temp in TEMPS:
            seqs = []
            tt = time.time()
            try:
                for _ in range(n_samp):
                    with torch.no_grad():
                        seq = model.sample(coords, args.gpu, temperature=temp)
                    seqs.append(seq)
            except Exception as e:
                print(f"  [FAIL] {stem} temp={temp}: {e}", flush=True)
                ok = False
                break
            per_temp_t[str(temp)] = round(time.time() - tt, 2)
            with open(os.path.join(OUT_DIR, f"{stem}.{temp}.seqs"), "w") as f:
                f.write("\n".join(seqs) + "\n")
        if not ok:
            stats["fail"] += 1
            continue
        stats["ok"] += 1
        n_new += 1
        if is_probe:
            probe_rec[stem] = {
                "len": len(native), "n_per_temp": n_samp,
                "wall_time_s": round(time.time() - t0, 2),
                "per_temp_s": per_temp_t,
                "peak_mem_MiB": round(torch.cuda.max_memory_allocated(args.gpu) / 2**20, 1),
            }
            json.dump({"note": "RhoDesign btrain_v2 单链采样效率（预测骨架批跑内测量；GPU1 4090）",
                       "gpu": torch.cuda.get_device_name(args.gpu),
                       "probes": probe_rec}, open(EFF_JSON, "w"), indent=1, ensure_ascii=False)
            print(f"  [PROBE] {stem} L={len(native)} {probe_rec[stem]['wall_time_s']}s "
                  f"peak={probe_rec[stem]['peak_mem_MiB']}MiB", flush=True)
        if (i + 1) % 10 == 0:
            print(f"进度 {i+1}/{len(todo)} (new {n_new}, skip {stats['skip']})", flush=True)
        if n_new % CHECKPOINT_EVERY == 0:
            s = write_metrics_json(stems, "checkpoint（批跑进行中）")
            print(f"  [ckpt] metrics 落盘: {s['coverage']}", flush=True)
    s = write_metrics_json(stems)
    print(f"完成: {stats} | {json.dumps({k: v for k, v in s.items() if k != 'note'}, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
