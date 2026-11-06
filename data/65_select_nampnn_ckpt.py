#!/usr/bin/env python3
"""第 65 步：NA-MPNN v2 重训 checkpoint 选择（val 集 teacher-forcing top-1 准确率）

na_run.py 只存 last.pt（每 epoch 覆盖）+ s_{step}.pt（每 1000 step）。
本脚本对全部候选 ckpt 在 val 96 链上算 top-1 准确率（与训练 valid_accuracy 同口径），
选最优作为 B 层推理权重。输出 survey/65_nampnn_ckpt_selection.json + 打印排序表。
用法：python data/65_select_nampnn_ckpt.py --gpu 0
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import pandas as pd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
NAMPNN_DIR = os.path.join(ROOT, "tools", "NA-MPNN")
CKPT_DIR = os.path.join(ROOT, "models", "NA-MPNN_btrain_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.chdir(NAMPNN_DIR)
    sys.path.insert(0, NAMPNN_DIR)
    device = torch.device(f"cuda:{args.gpu}")

    import cifutils, pdbutils
    from na_data_utils import PDBDataset
    from na_model_utils import featurize, ProteinMPNN, loss_nll

    params = json.load(open(os.path.join(DATA_DIR, "nampnn_btrain_v2", "nampnn_train_v2.json")))
    atom_list = ['N', 'CA', 'C', 'O',
                 'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
    cif_parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"], randomize_nmr_model=0)
    pdb_parser = pdbutils.PDBParser()
    ds = PDBDataset(cif_parser=cif_parser, pdb_parser=pdb_parser, atom_list_to_save=atom_list,
                    parse_protein=1, parse_dna=1, parse_rna=1, parse_rna_as_dna=0,
                    na_shared_tokens=params["NA_SHARED_TOKENS"],
                    protein_backbone_occ_cutoff=0.8, protein_side_chain_occ_cutoff=0.5,
                    dna_backbone_occ_cutoff=0.8, dna_side_chain_occ_cutoff=0.5,
                    rna_backbone_occ_cutoff=0.8, rna_side_chain_occ_cutoff=0.5,
                    crop_large_structures=0, batch_tokens=6000, na_ref_atom="C1'",
                    parse_ppms=0, min_overlap_length=5, drop_protein_probability=0,
                    na_only_as_uniform_ppm=0, protein_interface_residue_mutation_probability=0,
                    mutate_base_pair_together=0, mutate_entire_side_chain_interface_probability=0,
                    na_non_interface_as_uniform_ppm=0)

    df_valid = pd.read_csv(os.path.join(DATA_DIR, "nampnn_btrain_v2", "valid.csv"))
    df_valid["date"] = pd.to_datetime(df_valid["date"], format="%Y-%m-%d")

    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "s_*.pt")),
                   key=lambda p: int(os.path.basename(p)[2:-3])) + [os.path.join(CKPT_DIR, "last.pt")]
    ckpts = [c for c in ckpts if os.path.exists(c)]
    print(f"候选 ckpt: {[os.path.basename(c) for c in ckpts]}", flush=True)

    rows = []
    for ck in ckpts:
        model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                            num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
                            dropout=0.1, atom_dict=ds.atom_dict, restype_to_int=ds.restype_to_int,
                            polytype_to_int=ds.polytype_to_int, protein_augment_eps=0.0,
                            dna_augment_eps=0.0, rna_augment_eps=0.0, decode_protein_first=0,
                            na_ref_atom="C1'", include_pred_na_N=1, device=device,
                            vocab=33, num_letters=33)
        checkpoint = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        hit = tot = 0
        with torch.no_grad():
            for _, row in df_valid.iterrows():
                ex = row.to_dict()
                loaded = ds.loader(ex, "1")
                if loaded[0] == "pass":
                    continue
                out_dict, L = loaded
                # DataLoader default_collate 会把 numpy→tensor；手工复现（na_run.py 的 batch 格式）
                batch = [({k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                             for k, v in out_dict.items()},
                          torch.tensor(L, dtype=torch.int32))]
                fd = featurize(batch, ds.polytype_to_int, ds.restype_to_int, ds.atom_dict, device)
                if isinstance(fd, str):
                    continue
                log_probs, _ = model(fd)
                S = fd["S"]
                # 修复（2026-08-16）：featurize 无 chain_mask 键；改用 na_run.py 同口径
                # mask_for_loss = mask * S_mask（S_mask 剔除 UNK/DX/RX/MAS/PAD 无损失 token）
                tokens_with_no_loss = torch.tensor(
                    [ds.restype_to_int[t] for t in ("UNK", "DX", "RX", "MAS", "PAD")],
                    device=device)
                S_mask = 1 - (torch.any(S[:, :, None] == tokens_with_no_loss[None, None, :], dim=-1)).long()
                mask = fd["mask"] * S_mask
                pred = torch.argmax(log_probs, -1)
                # top-1 准确率（RNA 位、有效 mask）
                rna = fd["rna_mask"]
                m = (mask * rna).bool()
                hit += (pred[m] == S[m]).sum().item()
                tot += m.sum().item()
        acc = hit / max(1, tot)
        rows.append({"ckpt": os.path.basename(ck), "val_top1_acc": acc,
                     "step": checkpoint.get("step"), "epoch": checkpoint.get("epoch")})
        print(f"  {os.path.basename(ck)}: val top-1 acc = {acc:.4f} (epoch {checkpoint.get('epoch')})", flush=True)
        del model
        torch.cuda.empty_cache()
    rows.sort(key=lambda r: -r["val_top1_acc"])
    json.dump(rows, open(os.path.join(DATA_DIR, "survey", "65_nampnn_ckpt_selection.json"), "w"), indent=1)
    print(f"\n最优: {rows[0]['ckpt']} (val acc {rows[0]['val_top1_acc']:.4f})", flush=True)


if __name__ == "__main__":
    main()
