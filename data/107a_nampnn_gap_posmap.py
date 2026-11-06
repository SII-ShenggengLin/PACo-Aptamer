"""一次性 helper：对 NA-MPNN gap 链（61 号口径）跑官方 parse_PDB + difflib 等块对齐，
输出 {stem: pos_all}（pos_all[k] = 样本第 k 字符对应的 native 位置）。
复刻 data/61_compute_metrics_nampnn_btest.py 的 parse_native + align_positions。
运行：/root/miniconda3/envs/odesign_service_env/bin/python（prody 来自 data/survey/pylibs）
"""
import os, sys, json, difflib

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
sys.path.insert(0, os.getcwd())

restypes = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
            'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK',
            'DA','DC','DG','DT','DX','A','C','G','U','RX','MAS','PAD']
r31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','DA':'a','DC':'c','DG':'g',
       'DT':'t','DX':'x','A':'b','C':'d','G':'h','U':'u','RX':'y','MAS':'-','PAD':'+'}
alphabet = [r31[r] for r in restypes]
RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}

from data_utils import parse_PDB

canon = json.load(open(os.path.join(ROOT, "data/survey/canonical_v2.json")))
met = json.load(open(os.path.join(ROOT, "data/survey/metrics_NAMPNN_btest_v2.json")))
gaps = met["summary"]["gap_chains"]

out = {}
for stem in sorted(gaps):
    d, _, _, _, _ = parse_PDB(os.path.join(ROOT, "data/canonical_v2", stem + ".pdb"),
                              device="cpu", model_type="na_mpnn", parse_na_only=0,
                              na_shared_tokens=0, load_residues_with_missing_atoms=0)
    S = d["S"].cpu().numpy()
    parsed = "".join(RNA_MAP.get(alphabet[AA], "?") for AA in S)
    native = canon[stem]["seq"]
    sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
    pos = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pos.extend(range(i1, i2))
    assert len(pos) == gaps[stem], f"{stem}: len(pos_all)={len(pos)} != gap_len={gaps[stem]}"
    out[stem] = pos
    print(f"{stem}: parsed_len={len(parsed)} pos_all={len(pos)} native_len={len(native)}", flush=True)

fp = os.path.join(ROOT, "data/survey/107_nampnn_gap_posall.json")
json.dump(out, open(fp + ".tmp", "w"))
os.rename(fp + ".tmp", fp)
print("wrote", fp, len(out), "chains")
