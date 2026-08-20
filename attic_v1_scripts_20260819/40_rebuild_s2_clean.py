"""重建无泄漏 S2 训练数据：剔除与 blayer val/test 链序列相似度 ≥0.9 的链
用法：python 40_rebuild_s2_clean.py
产出：s2_btrain/s2_samples_clean.pt（575 链）+ 污染统计
"""
import json, torch
from difflib import SequenceMatcher
import sys
sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from TACoAptamer.data_utils import load_s1

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"

blayer = json.load(open(f"{DATA_DIR}/survey/blayer_splits.json"))
s2 = torch.load(f"{DATA_DIR}/s2_btrain/s2_samples.pt", map_location="cpu")
remove = set(json.load(open(f"{DATA_DIR}/s2_btrain/leak_remove.json")))

te_chains = set(k for k, v in blayer.items() if v == "test")
va_chains = set(k for k, v in blayer.items() if v == "val")

# 干净数据
s2_clean = {k: v for k, v in s2.items() if k not in remove}
torch.save(s2_clean, f"{DATA_DIR}/s2_btrain/s2_samples_clean.pt")
print(f"干净 S2 数据: {len(s2_clean)} 链（剔除 {len(s2) - len(s2_clean)}）")

# test 侧污染边界：488 条 test 链中多少与【原 S2 训练】同源 ≥0.9
test_data = load_s1(set(te_chains), cache=f"{DATA_DIR}/taco_test.pt")
def sim(a, b): return SequenceMatcher(None, a, b).ratio()

n_contam = 0
contam_chains = []
for b, td in test_data.items():
    hit = any(sim(d["sequence"], td["seq"]) >= 0.9 for d in s2.values())
    if hit:
        n_contam += 1
        contam_chains.append(b)
print(f"B-test {len(test_data)} 链中，与【原 S2 训练】同源 ≥0.9: {n_contam} 条（{n_contam/len(test_data)*100:.1f}%）")
with open(f"{DATA_DIR}/s2_btrain/test_contaminated.json", "w") as f:
    json.dump(sorted(contam_chains), f, indent=1)

# 同一统计针对干净数据（应降到 0）
n2 = sum(1 for b, td in test_data.items()
         if any(sim(d["sequence"], td["seq"]) >= 0.9 for d in s2_clean.values()))
print(f"B-test 与【干净 S2 训练】同源 ≥0.9: {n2}（应=0）")

# 界面链比例（干净数据）
n_iface = sum(1 for v in s2_clean.values() if (v["interface_mask"] > 0).sum() > 0 if hasattr(v["interface_mask"], "sum"))
print(f"干净数据含界面链: {n_iface}/{len(s2_clean)}")
