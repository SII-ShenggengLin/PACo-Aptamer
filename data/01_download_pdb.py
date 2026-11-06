#!/usr/bin/env python3
"""下载 BGSU 索引中训练窗+测试窗的 PDB 文件（AlignIF 复现第 1 步）"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
PDB_DIR = os.path.join(DATA_DIR, "rcsb_pdb")
os.makedirs(PDB_DIR, exist_ok=True)

bgsu = json.load(open("/tmp/bgsu_pdbs.json"))

# 时间窗：训练 ≤2024-05-01；独立测试 2024-05-01 → 2025-01-01
train_ids = [k for k, v in bgsu.items() if v.get("release_date", "9999") <= "2024-05-01"]
test_ids  = [k for k, v in bgsu.items() if "2024-05-01" < v.get("release_date", "0000") <= "2025-01-01"]
print(f"训练窗条目: {len(train_ids)}  测试窗条目: {len(test_ids)}", flush=True)

def download(pdb_id):
    out = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return (pdb_id, "cached")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "60", url, "-o", out],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000:
            return (pdb_id, "ok")
        time.sleep(2)
    return (pdb_id, "failed")

all_ids = train_ids + test_ids
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(download, pid): pid for pid in all_ids}
    for fut in as_completed(futs):
        pid, status = fut.result()
        done += 1
        if done % 200 == 0:
            print(f"进度: {done}/{len(all_ids)}", flush=True)

ok = sum(1 for f in os.listdir(PDB_DIR) if os.path.getsize(os.path.join(PDB_DIR, f)) > 1000)
print(f"完成。成功下载: {ok} / {len(all_ids)}")
