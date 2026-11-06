#!/usr/bin/env python3
"""第 78 步：AptaDB 爬取与解析（P1-1 回顾性验证，蓝图 §五 Fig6d）

AptaDB（lmmd.ecust.edu.cn/aptadb，RNA 2024 30(3):189）：1,350 条适配体-靶标交互、
1,293 序列、1,230 Kd。无批量下载/API → 礼貌爬取条目页 result.php?id=1..1350。

相位：
  --phase crawl   逐条抓取原始 HTML → survey/78_raw/{id}.html（可续爬，0.25s 限速）
  --phase parse   HTML → survey/78_aptadb.json（逐 id 记录 + 汇总统计）
  --phase stats   解析结果摸底：Kd 单位分布、靶标类型、RNA/DNA、家族规模

环境：/root/miniconda3/envs/aptamer_eval/bin/python（仅标准库）
"""
import os, re, json, time, argparse, urllib.request, html

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
RAW_DIR = os.path.join(SUR, "78_raw")
BASE = "https://lmmd.ecust.edu.cn/aptadb/result.php?id="
N_MAX = 1360   # 官方 1,350 交互，留少量余量探边界


def fetch(url, tries=3, timeout=20):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research crawl)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            if i == tries - 1:
                print(f"  [fail {url}: {e}]", flush=True)
                return None
            time.sleep(2)


def phase_crawl():
    os.makedirs(RAW_DIR, exist_ok=True)
    t0, n_new, n_skip, n_fail = time.time(), 0, 0, 0
    for i in range(1, N_MAX + 1):
        p = os.path.join(RAW_DIR, f"{i}.html")
        if os.path.exists(p) and os.path.getsize(p) > 5000:
            n_skip += 1
            continue
        h = fetch(BASE + str(i))
        if h is None or len(h) < 5000:
            n_fail += 1
            continue
        open(p, "w").write(h)
        n_new += 1
        if n_new % 100 == 0:
            print(f"[crawl] 新抓 {n_new}（跳过 {n_skip} 失败 {n_fail}）| {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.25)
    print(f"[crawl] 完成：新 {n_new} / 跳过 {n_skip} / 失败 {n_fail} | {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- parse
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<t([dh])[^>]*>(.*?)</t\1>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def clean(s):
    return html.unescape(TAG_RE.sub("", s)).strip()


def parse_one(path):
    h = open(path, encoding="utf-8", errors="replace").read()
    rec = {}
    rows = []
    for rm in ROW_RE.finditer(h):
        cells = [(t, clean(c)) for t, c in CELL_RE.findall(rm.group(1))]
        cells = [(t, c) for t, c in cells if c != ""]
        if cells:
            rows.append(cells)
    k = 0
    while k < len(rows):
        cells = rows[k]
        # 模式 A：表头行（全 th）+ 下一行（全 td 同长度）→ 按列 zip
        if all(t == "h" for t, _ in cells) and k + 1 < len(rows):
            nxt = rows[k + 1]
            if all(t == "d" for t, _ in nxt) and len(nxt) == len(cells):
                for (_, lab), (_, val) in zip(cells, nxt):
                    rec[lab] = val
                k += 2
                continue
        # 模式 B：行内标签-值交替（偶数个 cell）
        if len(cells) % 2 == 0:
            ok = True
            for j in range(0, len(cells), 2):
                if cells[j][0] == "d" and cells[j + 1][0] == "d":
                    continue
                ok = False
                break
            if ok and len(cells) >= 2:
                for j in range(0, len(cells), 2):
                    lab = cells[j][1].rstrip(":")
                    if lab:
                        rec.setdefault(lab, cells[j + 1][1])
        k += 1
    return rec


# 需要的字段 → 统一键名
WANT = {
    "Aptamer Sequence": "seq",
    "Aptamer ID": "aptamer_id",
    "Aptamer descriptor": "descriptor",
    "Target chemistry": "target_chemistry",
    "Target name": "target_name",
    "Affinity": "affinity",
    "Binding Conditions/Buffer": "buffer",
    "PubMed ID": "pubmed_id",
    "Aptamer chemistry": "chemistry",
    "Length": "length",
    "GC content": "gc",
    "Activity (Kd)": "activity_kd",
    "Assay": "assay",
    "Target type": "target_type",
    "Target unique ID": "target_uid",
    "Interaction ID": "interaction_id",
    "Description": "description",
}
SS_LABELS = ("optimal secondary structure", "centroid secondary structure")


def norm_rec(rec, iid):
    out = {"interaction_id": iid}
    for k, v in rec.items():
        if k in WANT:
            out[WANT[k]] = v
    # SS dot-bracket（按标签顺序在原始 HTML 里抓，配对语句更稳）
    h = open(os.path.join(RAW_DIR, f"{iid}.html"), encoding="utf-8", errors="replace").read()
    for lab, key in ((SS_LABELS[0], "ss_mfe"), (SS_LABELS[1], "ss_centroid")):
        m = re.search(re.escape(lab) + r"[^<]*</td>\s*<td[^>]*>([^<]+)</td>", h, re.I)
        if m:
            out[key] = clean(m.group(1))
    return out


def phase_parse():
    out = {}
    for i in range(1, N_MAX + 1):
        p = os.path.join(RAW_DIR, f"{i}.html")
        if not os.path.exists(p) or os.path.getsize(p) < 5000:
            continue
        rec = parse_one(p)
        out[i] = norm_rec(rec, i)
    json.dump(out, open(os.path.join(SUR, "78_aptadb.json"), "w"), indent=1, ensure_ascii=False)
    n = len(out)
    n_seq = sum(1 for v in out.values() if v.get("seq"))
    n_kd = sum(1 for v in out.values() if v.get("affinity"))
    print(f"[parse] 条目 {n} | 含序列 {n_seq} | 含 affinity {n_kd}")
    for i in list(out)[:3]:
        print(" ", json.dumps(out[i], ensure_ascii=False)[:300])


def phase_stats():
    out = json.load(open(os.path.join(SUR, "78_aptadb.json")))
    from collections import Counter
    chem = Counter(v.get("chemistry", "?") for v in out.values())
    ttype = Counter(v.get("target_chemistry") or v.get("target_type", "?") for v in out.values())
    # affinity 单位摸底
    units = Counter()
    noaff = 0
    for v in out.values():
        a = v.get("affinity") or ""
        m = re.search(r"(pM|nM|µM|uM|μM|mM|M)\s*$", a.strip())
        if m:
            units[m.group(1)] += 1
        elif a:
            units["<其他>:" + a[:20]] += 1
        else:
            noaff += 1
    # 每靶标家族规模（同 target_uid 的序列数）
    fam = Counter(v.get("target_uid") for v in out.values() if v.get("seq"))
    fam_dist = Counter(min(n, 10) for n in fam.values())
    print(f"chemistry: {dict(chem)}")
    print(f"target type: {dict(ttype)}")
    print(f"affinity 单位: {dict(units)} | 无 affinity: {noaff}")
    print(f"家族规模分布(10=≥10): {dict(sorted(fam_dist.items()))}")
    print(f"靶标数 {len(fam)} | ≥3 序列靶标 {sum(1 for n in fam.values() if n >= 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['crawl', 'parse', 'stats'])
    args = ap.parse_args()
    dict(crawl=phase_crawl, parse=phase_parse, stats=phase_stats)[args.phase]()


if __name__ == "__main__":
    main()
