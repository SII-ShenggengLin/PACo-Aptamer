#!/usr/bin/env python3
"""第 20 步：RhoDesign 批量逆折叠推理（官方权重 × 评估集）

- 模型：RhoDesignModel（GVP + Transformer），no_ss 权重（无需 SS 输入）
- 输入：grnade_inputs/{stem}.pdb（链级 PDB）
- 采样：temperature 0.1/0.5/1.0 × N 样本（长链降采样控制时间）
- 输出：survey/rhodesign_out/{stem}.{temp}.seqs + .native
- 环境：rhodesign_service_env（python 3.9, torch 1.12, torch_geometric 2.5 就绪）
- GPU：1、2
"""
import os, sys, json, argparse, random
import numpy as np
import torch

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoDesign/src")
from RhoDesign_without2d import RhoDesignModel   # no_ss 版本（对应 no_ss_apexp_best.pth）
from alphabet import Alphabet
from util import load_structure, extract_coords_from_structure

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
IN_DIR = os.path.join(DATA_DIR, "survey", "grnade_inputs_v2")
OUT_DIR = os.environ.get("RHO_OUT_DIR", os.path.join(DATA_DIR, "survey", "rhodesign_out_v2"))
os.makedirs(OUT_DIR, exist_ok=True)
# 本项目补丁：RHO_CKPT / RHO_OUT_DIR 环境变量覆盖（A 层重训评估复用本脚本）
CKPT = os.environ.get("RHO_CKPT", "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoDesign/checkpoint/no_ss_apexp_best.pth")
TEMPS = [0.1, 0.5, 1.0]
N_SAMPLES = 100

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

def main(gpu, task_list):
    random.seed(1)
    args = args_class()
    dictionary = Alphabet(["A", "G", "C", "U", "X"])
    model = RhoDesignModel(args, dictionary)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.cuda(gpu)
    model.eval()
    stats = {"ok": 0, "fail": 0}
    for pdb, chain in task_list:
        stem = f"{pdb}_{chain}"
        # 本项目补丁：RHOFIX_DIR 环境变量指定的修复目录优先（NMR 多模型/坏文件链）
        _fixdir = os.environ.get("RHOFIX_DIR")
        pdb_path = os.path.join(_fixdir, stem + ".pdb") if _fixdir else None
        if not pdb_path or not os.path.exists(pdb_path):
            pdb_path = os.path.join(IN_DIR, stem + ".pdb")
        outs = [os.path.join(OUT_DIR, f"{stem}.{t}.seqs") for t in TEMPS]
        native_p = os.path.join(OUT_DIR, f"{stem}.native")
        if all(os.path.exists(o) for o in outs):
            stats["ok"] += 1
            continue
        try:
            structure = load_structure(pdb_path)
            coords, native = extract_coords_from_structure(structure)
        except Exception as e:
            stats["fail"] += 1
            print(f"  [FAIL] {stem} load: {e}", flush=True)
            continue
        with open(native_p, "w") as f:
            f.write(native + "\n")
        n_samp = samples_for_len(len(native))
        for temp, out_f in zip(TEMPS, outs):
            seqs = []
            try:
                for _ in range(n_samp):
                    with torch.no_grad():
                        seq = model.sample(coords, gpu, temperature=temp)
                    seqs.append(seq)
            except Exception as e:
                print(f"  [FAIL] {stem} temp={temp}: {e}", flush=True)
                stats["fail"] += 1
                continue
            with open(out_f, "w") as f:
                f.write("\n".join(seqs) + "\n")
        stats["ok"] += 1
        print(f"  {stem}: ok ({len(native)}nt)", flush=True)
    print(f"[GPU{gpu}] 完成: {stats}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--taskfile", type=str, required=True)
    args = parser.parse_args()
    main(args.gpu, json.load(open(args.taskfile)))
