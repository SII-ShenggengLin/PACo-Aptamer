#!/usr/bin/env python3
"""第 46 步：PACoAptamer 案例研究评估（8HZD_A/B、8HZE_A）

背景：案例链（2024+ 新结构，case_frozen/independent_test）不在 B 层 test 477，
38 号评估不支持 A 层链 → 本脚本单独评估：
- 输入：grnade_inputs_v2/{stem}.pdb（43 号管线干净链文件）+ RNAfold SS 预测
  （与 45 号 chain_pdb 提取同款逻辑；8HZD/8HZE 不在 chain_pdb/chain_pdb_v2）
- 权重：models/PACoAptamer/s2_v2.pth（B 层 v2 重训，对案例链严格未见）
- 采样：3 温度 × 100 样本 = 300 池内 best（与主表同口径）
- 输出：survey/metrics_PACo_case.json
"""
import os, re, sys, json, subprocess, argparse
import numpy as np
import torch
import biotite.structure.io.pdb as pdbio

BASE = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA = os.path.join(BASE, "data")
sys.path.insert(0, os.path.join(BASE, "tools"))
from PACoAptamer.data_utils import to_paco9, ss_to_pairs
sys.path.insert(0, os.path.join(BASE, "data"))

G27 = ["P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
       "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
       "N6", "N7", "C8", "N9", "OP1", "OP2"]
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
TEMPS = [0.1, 0.5, 1.0]
STEMS = ["8HZD_A", "8HZD_B", "8HZE_A"]

CKPT = os.path.join(BASE, "models", "PACoAptamer", "s2_v2.pth")
IN_DIR = os.path.join(DATA, "survey", "grnade_inputs_v2")


def predict_ss(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None


def extract_pdb_stem(stem):
    """grnade_inputs_v2 链文件 → (seq, coords27)，45 号同款 P 原子代表"""
    f = os.path.join(IN_DIR, stem + ".pdb")
    if not os.path.exists(f):
        return None
    arr = pdbio.PDBFile.read(f).get_structure()[0]
    arr = arr[~arr.hetero]
    ap = arr[arr.atom_name == "P"]
    if ap.array_length() == 0:
        return None
    seq = []
    for rn in ap.res_name:
        rn = str(rn).strip().upper()
        b = "X"
        for x in "ACGU":
            if rn == x or rn.endswith(x):
                b = x
                break
        seq.append(b)
    s = "".join(seq)
    coords27 = np.full((len(s), 27, 3), 1e-5, dtype=np.float32)
    for i in range(len(s)):
        mask = (arr.res_id == ap.res_id[i])
        sub = arr[mask]
        for atom in sub.atom_name:
            a = str(atom)
            if a in G27:
                coords27[i, G27.index(a)] = sub.coord[sub.atom_name == atom][0]
    return s, coords27


def load_model(ckpt_path, device):
    # 38 号同款：args 容器 + 包导入（model.py 内部有相对导入，必须走包路径）
    sys.path.insert(0, os.path.join(BASE, "tools"))
    from PACoAptamer.model import PACoModel
    from PACoAptamer.alphabet import Alphabet
    args_cls = type('A', (), {'local_rank': -1, 'device_id': [0],
        'encoder_embed_dim': 512, 'decoder_embed_dim': 512, 'dropout': 0.1,
        'gvp_top_k_neighbors': 15, 'gvp_node_hidden_dim_vector': 256,
        'gvp_node_hidden_dim_scalar': 512, 'gvp_edge_hidden_dim_scalar': 32,
        'gvp_edge_hidden_dim_vector': 1, 'gvp_num_encoder_layers': 3,
        'gvp_dropout': 0.1, 'encoder_layers': 3, 'encoder_attention_heads': 4,
        'attention_dropout': 0.1, 'encoder_ffn_embed_dim': 512,
        'decoder_layers': 3, 'decoder_attention_heads': 4,
        'decoder_ffn_embed_dim': 512})()
    alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])
    model = PACoModel(args_cls, alphabet).cuda(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, alphabet


def sample_chain(model, alphabet, d, device, temperature, n_samples):
    """38 号同款：位置参数调用 model.sample(coords, ss, lig, ligm, conf, device, temperature)"""
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    lig = torch.zeros(1, 0, 21, device=device)
    ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
    seqs = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model.sample(coords, ss, lig, ligm, conf, device, temperature=temperature)
            seqs.append(''.join(alphabet.get_tok(t) for t in pred.tolist()))
    return seqs


def main():
    device = torch.device("cuda:0")
    model, alphabet = load_model(CKPT, device)
    results = {}
    for stem in STEMS:
        got = extract_pdb_stem(stem)
        if got is None:
            print(f"{stem}: 输入缺失", flush=True)
            continue
        s, coords27 = got
        ss = predict_ss(s)
        if ss is None or len(ss) != len(s):
            print(f"{stem}: SS 预测失败", flush=True)
            continue
        d = {"coords9": to_paco9(coords27), "ss_pairs": ss_to_pairs(ss), "seq": s}
        pool = []
        for t in TEMPS:
            pool.extend(sample_chain(model, alphabet, d, device, t, 100))
        recs = [sum(1 for a, b in zip(x, s) if a == b) / len(s) for x in pool]
        best = max(recs)
        t01 = [sum(1 for a, b in zip(x, s) if a == b) / len(s)
               for x in pool[:100]]
        results[stem] = {
            "len": len(s), "seq": s, "ss": ss,
            "recovery_best": best,
            "recovery_t01_mean": sum(t01) / len(t01),
            "n_pool": len(pool),
        }
        print(f"{stem}: best {best:.4f} | t0.1 均值 {results[stem]['recovery_t01_mean']:.4f} ({len(s)}nt)", flush=True)
    json.dump({"summary": {
        "n_chains": len(results),
        "recovery_best_mean": sum(r["recovery_best"] for r in results.values()) / len(results) if results else None,
    }, "per_chain": results},
        open(os.path.join(DATA, "survey", "metrics_PACo_case.json"), "w"), indent=1, ensure_ascii=False)
    print("完成 → survey/metrics_PACo_case.json", flush=True)


if __name__ == "__main__":
    main()
