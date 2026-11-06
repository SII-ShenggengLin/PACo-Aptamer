#!/usr/bin/env python3
"""第 99 步：PACo(ens2) 与 AlignIF 推理效率探针

6 条代表链（test 477 中长度最接近 50/100/150/200/250/300 nt 各一条）：
- PACo：标准池规模 300 样本/链（3 温度×100，ens2 逐步概率平均）；
  另测单模（seed43_noiface）同协议 → ensemble/单模开销比。
  墙钟、torch.cuda.max_memory_allocated 峰值、samples/sec、s/样本。
- AlignIF：20 样本/链（t=0.1，no-MStA，60 号同款子进程调用，输出到临时目录）；
  墙钟、吞吐；显存峰值由父进程线程轮询 torch.cuda.mem_get_info 的
  GPU 总占用增量近似（子进程独立 CUDA context，本进程计数器不可见）。

输出：survey/99_efficiency_PACoAptamer.json、survey/99_efficiency_AlignIF.json
用法：CUDA_VISIBLE_DEVICES=0 python data/99_efficiency_probe.py --which paco|alignif|all
"""
import os, sys, json, time, argparse, subprocess, threading
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100
TARGET_LENS = [50, 100, 150, 200, 250, 300]
ALIGNIF_DIR = os.path.join(ROOT, "tools", "AlignIF")
PY = "/root/miniconda3/envs/odesign_service_env/bin/python"
CKPT_AI = os.path.join(ROOT, "models", "AlignIF_btrain_v2", "alignif-seed=0-epoch=79.ckpt")

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m39 = _imp("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
from PACoAptamer.data_utils import load_s1


def pick_chains():
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test_stems = list(sp2["test"])
    data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    cands = [(s, len(data[s]['seq'])) for s in test_stems
             if s in data and len(data[s]['seq']) <= 512]
    picked = []
    for tl in TARGET_LENS:
        s, L = min(cands, key=lambda x: abs(x[1] - tl))
        picked.append((s, L))
        cands = [c for c in cands if c[0] != s]
    return picked, data


class GpuMemSampler:
    """轮询 GPU 总显存占用，取增量峰值（近似子进程显存）"""
    def __init__(self, interval=0.05):
        import torch
        self.interval = interval
        self.peak_delta = 0
        self._stop = threading.Event()
        self._base = None

    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self._base = torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]  # used
        self.peak_delta = 0

        def loop():
            import torch
            while not self._stop.is_set():
                used = torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]
                self.peak_delta = max(self.peak_delta, used - self._base)
                time.sleep(self.interval)
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join()
        return False


def probe_paco():
    import torch
    device = 0
    picked, data = pick_chains()
    print(f"[paco] 代表链: {picked}", flush=True)

    ens_models = []
    alphabet = None
    for n in ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]:
        m, alphabet = m74.load_model(os.path.join(ROOT, "models", "PACoAptamer", n + ".pth"), device)
        ens_models.append(m)
    single = ens_models[0]

    # warmup（不计时）：第一条链 10 样本
    d0 = data[picked[0][0]]
    m39.sample_chain_ens(ens_models, alphabet, d0, device, 0.5, 10)
    torch.cuda.synchronize()

    per_chain = {}
    for stem, L in picked:
        d = data[stem]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        for t in TEMPS:
            m39.sample_chain_ens(ens_models, alphabet, d, device, t, N_PER_TEMP)
        torch.cuda.synchronize()
        dt_ens = time.time() - t0
        mem_ens = torch.cuda.max_memory_allocated(device)

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        for t in TEMPS:
            m39.sample_chain_ens([single], alphabet, d, device, t, N_PER_TEMP)
        torch.cuda.synchronize()
        dt_s = time.time() - t0
        mem_s = torch.cuda.max_memory_allocated(device)

        per_chain[stem] = {
            "len": L, "n_samples": 300,
            "ens2": {"wall_s": dt_ens, "peak_mem_mb": mem_ens / 1e6,
                     "samples_per_s": 300 / dt_ens, "s_per_sample": dt_ens / 300},
            "single": {"wall_s": dt_s, "peak_mem_mb": mem_s / 1e6,
                       "samples_per_s": 300 / dt_s, "s_per_sample": dt_s / 300},
            "ens_overhead_ratio": dt_ens / dt_s,
        }
        print(f"[paco] {stem} L={L} ens2 {dt_ens:.2f}s/{mem_ens/1e6:.0f}MB "
              f"single {dt_s:.2f}s 开销比 {dt_ens/dt_s:.2f}", flush=True)

    out = {
        "protocol": {"pool": "3 温度×100=300 样本", "ens2": ["s2_v2_seed43_noiface",
                     "s2_v2_ssdo15_noiface"], "single": "s2_v2_seed43_noiface",
                     "feature": "load_s1 缓存（晶体骨架）", "warmup": "首链 10 样本"},
        "per_chain": per_chain,
        "summary": {
            "ens2_wall_s_mean": float(np.mean([v["ens2"]["wall_s"] for v in per_chain.values()])),
            "ens2_samples_per_s_mean": float(np.mean([v["ens2"]["samples_per_s"] for v in per_chain.values()])),
            "ens_overhead_ratio_mean": float(np.mean([v["ens_overhead_ratio"] for v in per_chain.values()])),
            "ens2_peak_mem_mb_max": float(max(v["ens2"]["peak_mem_mb"] for v in per_chain.values())),
        },
    }
    tmp = os.path.join(SUR, "99_efficiency_PACoAptamer.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUR, "99_efficiency_PACoAptamer.json"))
    print(json.dumps(out["summary"], indent=1), flush=True)


def probe_alignif():
    picked, data = pick_chains()
    print(f"[alignif] 代表链: {picked}", flush=True)
    out_dir = os.path.join(SUR, "99_eff_alignif_tmp")
    os.makedirs(out_dir, exist_ok=True)
    per_chain = {}
    for stem, L in picked:
        pdb = os.path.join(DATA_DIR, "canonical_v2", f"{stem}.pdb")
        cmd = [PY, "inference.py", "--pdbs_dir", os.path.join(DATA_DIR, "canonical_v2"),
               "--checkpoint", CKPT_AI, "--n_samples", "20", "--temperature", "0.1",
               "--use_msta", "False", "--output_dir", out_dir,
               "--input_structure_path", pdb]
        env = {**os.environ,
               "PYTHONPATH": os.path.join(DATA_DIR, "pylibs_alignif") + ":" + os.path.join(ROOT, "tools"),
               "CUDA_VISIBLE_DEVICES": "0", "NUPACKHOME": "/tmp"}
        with GpuMemSampler() as gs:
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ALIGNIF_DIR, capture_output=True, timeout=1800, env=env)
            dt = time.time() - t0
        fa = os.path.join(out_dir, stem, "out.fasta")
        n = 0
        if os.path.exists(fa):
            n = sum(1 for l in open(fa) if l.startswith(">id:"))
        per_chain[stem] = {
            "len": L, "n_samples": n, "wall_s": dt,
            "peak_mem_mb_approx": gs.peak_delta / 1e6,
            "samples_per_s": n / dt if dt > 0 else None,
            "s_per_sample": dt / n if n else None,
            "rc": r.returncode,
        }
        print(f"[alignif] {stem} L={L} {dt:.1f}s n={n} peak≈{gs.peak_delta/1e6:.0f}MB", flush=True)
    out = {
        "protocol": {"pool": "20 样本 t=0.1 no-MStA", "ckpt": os.path.basename(CKPT_AI),
                     "input": "canonical_v2 晶体 PDB",
                     "mem_note": "父进程线程轮询 GPU 总显存增量近似（子进程独立 context）"},
        "per_chain": per_chain,
        "summary": {
            "wall_s_mean": float(np.mean([v["wall_s"] for v in per_chain.values()])),
            "samples_per_s_mean": float(np.mean([v["samples_per_s"] for v in per_chain.values()
                                                 if v["samples_per_s"]])),
            "s_per_sample_mean": float(np.mean([v["s_per_sample"] for v in per_chain.values()
                                                if v["s_per_sample"]])),
            "peak_mem_mb_approx_max": float(max(v["peak_mem_mb_approx"] for v in per_chain.values())),
        },
    }
    tmp = os.path.join(SUR, "99_efficiency_AlignIF.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUR, "99_efficiency_AlignIF.json"))
    print(json.dumps(out["summary"], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["paco", "alignif", "all"])
    args = ap.parse_args()
    if args.which in ("paco", "all"):
        probe_paco()
    if args.which in ("alignif", "all"):
        probe_alignif()
