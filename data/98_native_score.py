#!/usr/bin/env python3
"""第 98 步：native 序列 teacher-forced 打分（perplexity 主表列 + 逐样本 logP）

口径：AlignIF 论文——模型对 native 序列的逐位平均对数概率（自然对数），
ppl_native = exp(-mean logP/nt)；非 ACGU 位（canonical 的 'N'）用占位 token 喂入、剔除出平均。
同时输出 t=0.1 采样池逐样本 mean logP/nt（同一有效位掩码），供跨方法置信度分析。

- gRNAde：ckpt models/gRNAde_btrain_v2/best_checkpoint.h5；输入三件套
  survey/grnade_inputs_btest_v2/{stem}.pdb/.seq/.ss；token 序 = 官方 featurizer
  {A:0,G:1,C:2,U:3}（注意：26v2 推理脚本解码池用的 ["A","C","G","U"] 与之 C/G 互换，
  打分必须用官方序）。
- NA-MPNN：run.py 无 score 模式 → 手写 teacher-forced：parse_PDB + featurize +
  model.sample 官方路径，chain_mask=1（记录 log_probs）+ bias 钉死（真值位 0、其余 -1e9）
  强制逐位解码出 native/样本 token（S_t 被钉死为真值，条件上下文即真值前缀）。
  PDB 输入 = data/canonical_v2/{stem}.pdb（与 btest 推理同输入）。
- RhoDesign：teacher-forced forward（27v2 训练同款）；PDB 解析顺序
  rhodesign_btest_fix4 → chain_pdb → chain_pdb_v2（与 29v2 btest 一致）；
  native = survey/rhodesign_out_btest_v2/{stem}.native（29v2 落盘，与坐标同源）；
  非 ACGU→X 位剔除出平均。

环境（aptamer_eval 已删除）：
  gRNAde/NA-MPNN: /root/miniconda3/envs/odesign_service_env/bin/python
                  PYTHONPATH 分别需 data/survey/pyshims（cpdb stub）/ data/survey/pylibs（prody）
  RhoDesign:      /root/miniconda3/envs/rhodesign_service_env/bin/python
用法：
  CUDA_VISIBLE_DEVICES=2 <py> data/98_native_score.py --method grnade|nampnn|rhodesign [--max-minutes 40]
  汇总: <py> data/98_native_score.py --merge
输出：survey/98_part_{method}.json（断点续跑）→ survey/98_perplexity.json
"""
import os, sys, json, math, time, argparse

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")

TEST = sorted(json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))["test"])
CANON = json.load(open(os.path.join(SUR, "canonical_v2.json")))


def save_json(obj, fp):
    tmp = fp + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False)
    os.rename(tmp, fp)


def load_part(method):
    fp = os.path.join(SUR, f"98_part_{method}.json")
    return json.load(open(fp)) if os.path.exists(fp) else {}


def read_lines(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


# ================= gRNAde =================

def run_grnade(gpu, max_minutes, limit=0):
    os.environ.setdefault("X3DNA", "/tmp/x3dna")
    os.environ.setdefault("ETERNAFOLD", "/tmp/eternafold")
    os.environ.setdefault("PROJECT_PATH", os.path.join(ROOT, "tools", "gRNAde"))
    sys.path.insert(0, os.path.join(ROOT, "tools", "gRNAde"))
    sys.path.insert(0, os.path.join(SUR, "pyshims"))  # cpdb stub（featurize_custom 路径不调用）
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from src.models import gRNAde
    from src.data.featurizer import RNAGraphFeaturizer

    IN_DIR = os.path.join(SUR, "grnade_inputs_btest_v2")
    OUT_DIR = os.path.join(SUR, "grnade_out_btest_v2")
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
        return featurizer.featurize(rna)

    def mean_logp(batch, tok, valid):
        batch.seq = tok
        with torch.no_grad():
            logits = model(batch)
        lp = F.log_softmax(logits, dim=-1)
        per = lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return per[valid].mean().item()

    part = load_part("grnade")
    t0 = time.time()
    n_done = 0
    for stem in (TEST[:limit] if limit else TEST):
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
            # t=0.1 池逐样本（长度过滤，同一有效位掩码）
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
                    logits = model(bb)  # [B*L, 4]
                lp = F.log_softmax(logits, dim=-1).view(len(chunk), L, 4)
                tok_b = torch.stack([g.seq for g in graphs])  # [B, L]
                per = lp.gather(-1, tok_b.unsqueeze(-1)).squeeze(-1)  # [B, L]
                sample_logp.extend(per[:, valid].mean(-1).tolist())
            part[stem] = {"logp_native": lp_nat, "ppl_native": math.exp(-lp_nat),
                          "n_pos": int(valid.sum()), "len": L,
                          "sample_logp_t01": sample_logp}
            n_done += 1
            if n_done % 25 == 0:
                save_json(part, os.path.join(SUR, "98_part_grnade.json"))
                print(f"  [grnade] {len(part)}/{len(TEST)} ppl~{math.exp(-lp_nat):.2f}", flush=True)
            if (time.time() - t0) / 60 > max_minutes:
                print("  [grnade] 达时间上限，落盘退出", flush=True)
                break
        except Exception as e:
            part[stem] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {stem}: {e}", flush=True)
    save_json(part, os.path.join(SUR, "98_part_grnade.json"))
    print(f"[grnade] 完成 {sum(1 for v in part.values() if 'logp_native' in v)}/{len(TEST)}", flush=True)


# ================= NA-MPNN =================

def run_nampnn(gpu, max_minutes, limit=0):
    os.environ["PYTHONPATH"] = os.path.join(SUR, "pylibs")
    sys.path.insert(0, os.path.join(SUR, "pylibs"))
    NP_DIR = os.path.join(ROOT, "tools", "NA-MPNN", "inference")
    os.chdir(NP_DIR)
    sys.path.insert(0, NP_DIR)
    import torch
    from data_utils import parse_PDB, featurize
    from model_utils import ProteinMPNN

    atom_types = ['N', 'CA', 'C', 'O', 'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'",
                  "C3'", "O3'", "C2'", "O2'", "C1'"]
    atom_dict = dict(zip(atom_types, range(len(atom_types))))
    polytypes = ['PP', 'DNA', 'RNA', 'UNK', 'MAS', 'PAD']
    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))
    restypes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                'UNK', 'DA', 'DC', 'DG', 'DT', 'DX', 'A', 'C', 'G', 'U', 'RX', 'MAS', 'PAD']
    restype_to_int = dict(zip(restypes, range(len(restypes))))
    # na_shared_tokens=1（59 号推理默认）：RNA 字母共享 DNA token int
    restype_to_int["A"] = restype_to_int["DA"]
    restype_to_int["C"] = restype_to_int["DC"]
    restype_to_int["G"] = restype_to_int["DG"]
    restype_to_int["U"] = restype_to_int["DT"]
    restype_to_int["RX"] = restype_to_int["DX"]
    NUM_LETTERS = 33
    TOK = {c: restype_to_int[c] for c in "ACGU"}  # native/样本字母 → token int

    device = torch.device(f"cuda:{gpu}")
    ckpt = torch.load(os.path.join(ROOT, "models", "NA-MPNN_btrain_v2", "s_11016.pt"),
                      map_location=device, weights_only=False)
    model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
                        model_type="na_mpnn", vocab=33, num_letters=NUM_LETTERS,
                        atom_dict=atom_dict, restype_to_int=restype_to_int,
                        polytype_to_int=polytype_to_int)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    CANON_DIR = os.path.join(DATA_DIR, "canonical_v2")
    OUT_DIR = os.path.join(SUR, "nampnn_out_btest_v2")

    def score_seqs(feature_dict, L, seqs_tokens):
        """seqs_tokens: list of LongTensor[L]；bias 钉死逐位 token → teacher-forced logP。
        返回每条序列的逐位 logP [L]。"""
        outs = []
        for tok in seqs_tokens:
            feature_dict["S"] = tok[None, :]
            feature_dict["randn"] = torch.randn([1, L], device=device)
            # bias 钉死：真值位 0，其余 -1e9 → multinomial 必取真值 token
            feature_dict["bias"] = torch.full([1, L, NUM_LETTERS], -1e9, device=device)
            feature_dict["bias"][0, torch.arange(L), tok] = 0.0
            with torch.no_grad():
                out = model.sample(feature_dict)
            lp = out["log_probs"][0]  # [L, 33]
            got = out["S"][0]
            assert torch.equal(got, tok), "bias 钉死失败"
            outs.append(lp.gather(-1, tok[:, None]).squeeze(-1))  # [L]
        return outs

    part = load_part("nampnn")
    t0 = time.time()
    n_done = 0
    for stem in (TEST[:limit] if limit else TEST):
        if stem in part and "logp_native" in part[stem]:
            continue
        try:
            pdb = os.path.join(CANON_DIR, f"{stem}.pdb")
            md, _, _, _, _ = parse_PDB(pdb, device=device, chains=[],
                                       model_type="na_mpnn", parse_na_only=0,
                                       na_shared_tokens=1,
                                       load_residues_with_missing_atoms=0)
            md["chain_mask"] = torch.ones_like(md["S"])  # 全 design：记录 log_probs
            fd = featurize(md)
            fd["batch_size"] = 1
            fd["temperature"] = 1.0
            fd["symmetry_residues"] = [[]]
            fd["symmetry_weights"] = [[]]
            L = fd["S"].shape[1]
            native = CANON[stem]["seq"]
            S_np = fd["S"][0].cpu().numpy()
            r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                   'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                   'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X',
                   'DA':'a','DC':'c','DG':'g','DT':'t','DX':'x','A':'b','C':'d','G':'h',
                   'U':'u','RX':'y','MAS':'-','PAD':'+'}
            alpha = [r31[r] for r in restypes]
            RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U",
                       "a": "A", "c": "C", "g": "G", "t": "U"}  # shared_tokens 下 RNA 走 DNA int
            parsed = "".join(RNA_MAP.get(alpha[t], "?") for t in S_np)
            # 解析序列 ↔ canonical native 对齐（缺 C1' 残基被 parse 丢弃 / 修饰残基替换的链）
            if parsed == native:
                pos_map = list(range(L))  # parsed i ↔ native i
            else:
                import difflib
                sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
                pos_map = [None] * L
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        for k in range(i2 - i1):
                            pos_map[j1 + k] = i1 + k
            # 有效位：可映射且 parsed 字母 == native 字母 ∈ ACGU（同 61 号有效位口径）
            valid_parsed = [j for j in range(L)
                            if pos_map[j] is not None and parsed[j] != "?"
                            and parsed[j] == native[pos_map[j]] and parsed[j] in "ACGU"]
            if not valid_parsed:
                raise ValueError("无有效位")
            # 无效位（native 'N' 或 parse 未知/修饰残基）钉 TOK["A"] 占位：
            # UNK/DX/RX/MAS/PAD 行在 sample() 中被置零，钉它们会让 multinomial 拿到全零行（nan）
            nat_tok = torch.tensor([TOK[native[pos_map[j]]]
                                    if (pos_map[j] is not None and native[pos_map[j]] in "ACGU")
                                    else TOK["A"] for j in range(L)],
                                   dtype=torch.long, device=device)
            # native 打分：钉死 canonical native token（而非 parse 出的 S——二者 ACGU 位一致）
            seqs_t = [nat_tok]
            pool = [s for s in read_lines(os.path.join(OUT_DIR, f"{stem}.seqs"))
                    if len(s) == L]
            for s in pool:
                seqs_t.append(torch.tensor([TOK[c] for c in s], dtype=torch.long, device=device))
            lps = score_seqs(fd, L, seqs_t)
            vmask = torch.tensor([j in set(valid_parsed) for j in range(L)], device=device)
            lp_nat = lps[0][vmask].mean().item()
            part[stem] = {"logp_native": lp_nat, "ppl_native": math.exp(-lp_nat),
                          "n_pos": int(vmask.sum()), "len": len(native),
                          "sample_logp_t01": [lp[vmask].mean().item() for lp in lps[1:]]}
            n_done += 1
            if n_done % 10 == 0:
                save_json(part, os.path.join(SUR, "98_part_nampnn.json"))
                print(f"  [nampnn] {len(part)}/{len(TEST)}", flush=True)
            if (time.time() - t0) / 60 > max_minutes:
                print("  [nampnn] 达时间上限，落盘退出", flush=True)
                break
        except Exception as e:
            part[stem] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
    save_json(part, os.path.join(SUR, "98_part_nampnn.json"))
    print(f"[nampnn] 完成 {sum(1 for v in part.values() if 'logp_native' in v)}/{len(TEST)}", flush=True)


# ================= RhoDesign =================

def run_rhodesign(gpu, max_minutes, limit=0):
    sys.path.insert(0, os.path.join(ROOT, "tools", "RhoDesign_official", "src"))
    import torch
    import torch.nn.functional as F
    from RhoDesign_without2d import RhoDesignModel
    from alphabet import Alphabet
    from util import load_structure, extract_coords_from_structure, CoordBatchConverter

    class args_class:
        def __init__(self):
            self.local_rank = -1
            self.device_id = [0]
            self.epochs = 30; self.lr = 1e-4; self.batch_size = 1
            self.encoder_embed_dim = 512; self.decoder_embed_dim = 512; self.dropout = 0.1
            self.gvp_top_k_neighbors = 15
            self.gvp_node_hidden_dim_vector = 256; self.gvp_node_hidden_dim_scalar = 512
            self.gvp_edge_hidden_dim_scalar = 32; self.gvp_edge_hidden_dim_vector = 1
            self.gvp_num_encoder_layers = 3; self.gvp_dropout = 0.1
            self.encoder_layers = 3; self.encoder_attention_heads = 4
            self.attention_dropout = 0.1; self.encoder_ffn_embed_dim = 512
            self.decoder_layers = 3; self.decoder_attention_heads = 4
            self.decoder_ffn_embed_dim = 512

    device = 0  # CUDA_VISIBLE_DEVICES 限制后从 0 数
    args = args_class()
    dictionary = Alphabet(["A", "G", "C", "U", "X"])
    model = RhoDesignModel(args, dictionary)
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "RhoDesign_btrain_v2", "best_val.pth"),
                                     map_location="cpu"))
    model = model.cuda(device).eval()
    batch_converter = CoordBatchConverter(dictionary)
    pad_idx = dictionary.padding_idx
    x_idx = dictionary.get_idx("X")

    FIX4 = os.path.join(SUR, "rhodesign_btest_fix4")
    DIRS = [FIX4, os.path.join(DATA_DIR, "chain_pdb"), os.path.join(DATA_DIR, "chain_pdb_v2")]
    OUT_DIR = os.path.join(SUR, "rhodesign_out_btest_v2")

    def norm_seq(s):
        return "".join({"I": "G"}.get(c, c if c in "ACGU" else "X") for c in s)

    def score_seq(coords, seq_str):
        """teacher-forced mean logP/nt（忽略 pad 与 X 目标位）"""
        bc, conf, _, tokens, pmask, _ = batch_converter([(coords, None, seq_str, None)])
        tokens = tokens.cuda(device)
        pmask = pmask.bool().cuda(device)
        bc = bc.cuda(device)
        conf = conf.cuda(device)
        with torch.no_grad():
            logits, _ = model(bc, bc, pmask, conf, prev_output_tokens=tokens[:, :-1])
        logits = logits.permute(0, 2, 1)  # (B, T, V)
        lp = F.log_softmax(logits, dim=-1)
        tgt = tokens[:, 1:]
        per = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # [T]
        valid = (tgt[0] != pad_idx) & (tgt[0] != x_idx)
        return per[valid].mean().item(), int(valid.sum())

    part = load_part("rhodesign")
    t0 = time.time()
    n_done = 0
    for stem in (TEST[:limit] if limit else TEST):
        if stem in part and "logp_native" in part[stem]:
            continue
        try:
            pdb = None
            for d in DIRS:
                p = os.path.join(d, stem + ".pdb")
                if os.path.exists(p):
                    pdb = p
                    break
            if pdb is None:
                raise ValueError("no_pdb")
            structure = load_structure(pdb)
            coords, native_raw = extract_coords_from_structure(structure)
            native = norm_seq(native_raw)
            lp_nat, npos = score_seq(coords, native)
            pool = [norm_seq(s) for s in read_lines(os.path.join(OUT_DIR, f"{stem}.0.1.seqs"))
                    if len(s) == len(native)]
            sample_logp = [score_seq(coords, s)[0] for s in pool]
            part[stem] = {"logp_native": lp_nat, "ppl_native": math.exp(-lp_nat),
                          "n_pos": npos, "len": len(native),
                          "sample_logp_t01": sample_logp}
            n_done += 1
            if n_done % 25 == 0:
                save_json(part, os.path.join(SUR, "98_part_rhodesign.json"))
                print(f"  [rhodesign] {len(part)}/{len(TEST)}", flush=True)
            if (time.time() - t0) / 60 > max_minutes:
                print("  [rhodesign] 达时间上限，落盘退出", flush=True)
                break
        except Exception as e:
            part[stem] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
    save_json(part, os.path.join(SUR, "98_part_rhodesign.json"))
    print(f"[rhodesign] 完成 {sum(1 for v in part.values() if 'logp_native' in v)}/{len(TEST)}", flush=True)


# ================= 汇总 =================

def merge():
    import numpy as np
    out = {}
    macro = {}
    for method, tag in [("gRNAde", "grnade"), ("NA-MPNN", "nampnn"), ("RhoDesign", "rhodesign")]:
        part = load_part(tag)
        ok = {s: v for s, v in part.items() if "logp_native" in v}
        if not ok:
            continue
        out[method] = ok
        ppls = [v["ppl_native"] for v in ok.values()]
        macro[method] = {
            "n_chains": len(ok),
            "ppl_native_macro_mean": float(np.mean(ppls)),
            "logp_native_macro_mean": float(np.mean([v["logp_native"] for v in ok.values()])),
            "sample_logp_t01_macro_mean": float(np.mean(
                [np.mean(v["sample_logp_t01"]) for v in ok.values() if v["sample_logp_t01"]])),
            "n_error": len(part) - len(ok),
        }
    out["macro_summary"] = macro
    save_json(out, os.path.join(SUR, "98_perplexity.json"))
    print(json.dumps(macro, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default=None,
                    choices=["grnade", "nampnn", "rhodesign"])
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA_VISIBLE_DEVICES 限制后的逻辑设备号")
    ap.add_argument("--max-minutes", type=float, default=40)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.method == "grnade":
        run_grnade(args.gpu, args.max_minutes, args.limit)
    elif args.method == "nampnn":
        run_nampnn(args.gpu, args.max_minutes, args.limit)
    elif args.method == "rhodesign":
        run_rhodesign(args.gpu, args.max_minutes, args.limit)
