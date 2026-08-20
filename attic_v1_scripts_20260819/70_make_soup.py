#!/usr/bin/env python3
"""第 70 步：模型 soup（权重平均）构造——多 S2 变体 state_dict 算术平均

- 用法：python data/70_make_soup.py --ckpts s2_v2_batch4,s2_v2_batch4r --out soup_bb
- 输出：models/TACoAptamer/{out}.pth（同 38 号可直接加载评估）
- 注意：跨随机种子的 S1 重训（如 seed43）处于不同损失盆地，soup 可能失败——
  按"同盆地为先"原则优先平均同源 ckpt（同一 S1 init 的多次 S2、或同 seed42 系）。
"""
import os, argparse, torch

CKPT_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/TACoAptamer"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True, help="逗号分隔 ckpt 名（不带 .pth）")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    names = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    avg = None
    for n in names:
        sd = torch.load(os.path.join(CKPT_DIR, n + ".pth"), map_location="cpu")
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(names)
    out_path = os.path.join(CKPT_DIR, args.out + ".pth")
    torch.save(avg, out_path)
    print(f"soup({len(names)}): {'+'.join(names)} -> {out_path}")

if __name__ == "__main__":
    main()
