"""PACoAptamer val 恢复率评估（t=0.1 单样本，无配体输入口径，与 train.py eval_recovery 一致）
用法：CUDA_VISIBLE_DEVICES=1 python 39_val_recovery.py --ckpt <路径> [--gpu 1]
输出：stdout 打印 val recovery
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from PACoAptamer.model import PACoModel
from PACoAptamer.alphabet import Alphabet
from PACoAptamer.data_utils import load_s1

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    device = 0   # CUDA_VISIBLE_DEVICES 限制后从 0 数

    class args_class:
        def __init__(self):
            self.local_rank = -1
            self.device_id = [0]
            self.lr = 1e-4
            self.encoder_embed_dim = 512
            self.decoder_embed_dim = 512
            self.dropout = 0.1
            self.gvp_top_k_neighbors = 15
            self.gvp_node_hidden_dim_vector = 256
            self.gvp_node_hidden_dim_scalar = 512
            self.gvp_edge_hidden_dim_scalar = 32
            self.gvp_edge_hidden_dim_vector = 1
            self.gvp_num_encoder_layers = 3
            self.gvp_dropout = 0.1
            self.encoder_layers = 3
            self.encoder_attention_heads = 4
            self.attention_dropout = 0.1
            self.encoder_ffn_embed_dim = 512
            self.decoder_layers = 3
            self.decoder_attention_heads = 4
            self.decoder_ffn_embed_dim = 512

    model_args = args_class()
    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = PACoModel(model_args, alphabet).cuda(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
    val_stems = [k for k, v in blayer.items() if v == "val"]
    val_data = load_s1(set(val_stems), cache=os.path.join(DATA_DIR, "paco_val.pt"))
    val_ok = [s for s in val_stems if s in val_data]
    print(f"val 链 {len(val_ok)}", flush=True)

    recs = []
    with torch.no_grad():
        for stem in val_ok:
            d = val_data[stem]
            L = len(d["seq"])
            coords = torch.tensor(d["coords9"], dtype=torch.float32, device=device).unsqueeze(0)
            ss = torch.tensor(d["ss_pairs"], dtype=torch.bool, device=device).unsqueeze(0)
            confidence = torch.ones(1, L, device=device)
            lig_feats = torch.zeros(1, 0, 21, device=device)
            lig_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
            pred = model.sample(coords, ss, lig_feats, lig_mask, confidence, device,
                                temperature=0.1)
            rec = sum(1 for a, b in zip(pred.tolist(), d["seq"])
                      if alphabet.get_idx(b) == a) / max(1, len(d["seq"]))
            recs.append(rec)
    print(f"val recovery: {sum(recs)/len(recs):.4f}", flush=True)


if __name__ == "__main__":
    main()
