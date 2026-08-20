# -*- coding: utf-8 -*-
"""Fig 6a 画廊 PyMOL 渲染（12 案 × 回折-天然叠加）。

运行（无头、后台）:
    /root/miniconda3/envs/egarps_env/bin/pymol -cq fig6_gallery_pymol.py

每格内容（与 NACraft Fig3 同款四色语义，映射到本项目色板）:
  天然链   natv 灰绿 cartoon, 半透明 0.45（衬底）
  TACo 回折 紫色 cartoon（主叙述层, RhoFold+ 预测, align 到天然链）
  界面残基  ifblue 雾蓝段（<5 Å 配体的天然残基, 按序号映射染在回折链上）
  配体    ligor 橙色 sticks
产出: out/fig6_gallery/{stem}.png (1200×800, 透明底, ray)
诊断: 每链打印 天然残基数 / 回折残基数 / 界面残基序号（顺序索引）
"""
import os

from pymol import cmd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
OUT = os.path.join(ROOT, "figures", "out", "fig6_gallery")
os.makedirs(OUT, exist_ok=True)

# stem, 配体 resn, 排除的 HETATM（离子/多胺/共晶添加剂）
PICKS = [
    ("3GAO_A", "XAN", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("6UC7_B", "Q44", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("7Q80_E", "GUN", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD+1MA"),
    ("3FU2_C", "PRF", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("8D5L_A", "QIJ", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("6UET_A", "SAM", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("5BTP_B", "AMZ", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("8HB1_A", "NMN", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD+CBV"),
    ("4K31_B", "AM2", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("4FE5_B", "HPA", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("6PRV_D", "GTP", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
    ("6E8S_B", "EKJ", "HOH+MG+NA+K+CL+MN+ZN+CA+SO4+PO4+GOL+EDO+SPM+PUT+SPD"),
]

# 家族统一朝向:成员 → 参考链(参考须在 PICKS 中先渲染)
FAMILY_REF = {
    "6UC7_B": "3GAO_A",   # 嘌呤核糖开关三联:guanine riboswitch aptamer domain
    "4FE5_B": "3GAO_A",
}
_ref_views = {}

def define_colors():
    """reinitialize 会清掉自定义颜色,须在每格循环内重定义。"""
    cmd.set_color("natv",  [0x7F / 255, 0xA8 / 255, 0x8B / 255])   # 天然灰绿
    cmd.set_color("ifblue", [0x6C / 255, 0x9B / 255, 0xD1 / 255])  # 界面雾蓝
    cmd.set_color("desp",  [0x9B / 255, 0x7E / 255, 0xB8 / 255])   # 回折紫
    cmd.set_color("ligor", [0xE8 / 255, 0xA0 / 255, 0x88 / 255])   # 配体橙

cmd.set("ray_opaque_background", "off")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.set("cartoon_sampling", 11)
cmd.set("cartoon_smooth_cycles", 2)
cmd.set("cartoon_tube_radius", 0.38)
cmd.set("cartoon_tube_cap", 2)
cmd.set("orthoscopic", "on")


def residues_in_order(sel):
    seen, order = set(), []
    for a in cmd.get_model(sel).atom:
        key = (a.chain, a.resi_number)
        if key not in seen:
            seen.add(key)
            order.append(a.resi_number)
    return order


def residue_letters(sel):
    """按顺序返回残基单字母(修饰核苷酸归并为母体碱基)。"""
    seen, letters = set(), []
    mod = {"1MA": "A", "5MC": "C", "PSU": "U", "OMC": "C", "5BU": "U",
           "2MG": "G", "7DA": "A", "H2U": "U", "DU": "U", "1MG": "G",
           "M2G": "G", "OMG": "G", "FHU": "U", "A2M": "A", "3TD": "U"}
    for a in cmd.get_model(sel).atom:
        key = (a.chain, a.resi_number)
        if key not in seen:
            seen.add(key)
            rn = a.resn.strip().upper()
            letters.append(mod.get(rn, rn if rn in "ACGU" else "N"))
    return letters


def nw_map(a_seq, b_seq):
    """Needleman-Wunsch:返回 a 序列每个位置对应 b 序列的位置(无对应为 None)。"""
    GAP, MATCH, MISMATCH = -1, 1, -1
    n, m = len(a_seq), len(b_seq)
    F = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        F[i][0] = i * GAP
    for j in range(1, m + 1):
        F[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = MATCH if a_seq[i - 1] == b_seq[j - 1] else MISMATCH
            F[i][j] = max(F[i - 1][j - 1] + d, F[i - 1][j] + GAP,
                          F[i][j - 1] + GAP)
    mapping, i, j = [None] * n, n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and
                F[i][j] == F[i - 1][j - 1] +
                (MATCH if a_seq[i - 1] == b_seq[j - 1] else MISMATCH)):
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and F[i][j] == F[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    return mapping


for stem, lig_resn, excl in PICKS:
    pdbid, ch = stem.split("_")
    cmd.reinitialize()
    define_colors()
    cmd.set("ray_opaque_background", "off")
    cmd.set("ray_shadows", 0)
    cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", f"{pdbid}.pdb"), "cx")

    # 天然链 = 该链中含 C4' 原子的残基整体(核酸及修饰核苷酸,排除蛋白/辅料)
    cmd.select("nat", f"byres (cx and chain {ch} and name C4')")

    # 家族统一朝向:同家族成员(嘌呤核糖开关 3GAO/6UC7/4FE5)把整个复合物
    # 叠加到参考(3GAO_A)天然链坐标系,相机复用参考 view → 网格内同折同视角。
    ref_stem = FAMILY_REF.get(stem)
    if ref_stem:
        refpdb, refch = ref_stem.split("_")
        cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", f"{refpdb}.pdb"), "refcx")
        cmd.select("refnat", f"byres (refcx and chain {refch} and name C4')")
        # align(mobile, target): mobile=nat 属 cx 对象 → cx(含配体)整体被搬运
        rms = cmd.align("nat and name C4'", "refnat and name C4'", cycles=8)
        cmd.delete("refcx")
        print(f"[{stem}] family-align onto {ref_stem}: rms={rms[0]:.2f}Å "
              f"n={rms[1]}")

    cmd.load(os.path.join(ROOT, "data", "survey", "86_sctm", f"TACo_{stem}.pdb"), "des")
    # 仅保留距该链 5 Å 内的配体拷贝
    cmd.select("lig", f"byres ((cx and resn {lig_resn}) within 5.0 of nat)")
    cmd.select("desp", "des and not resn HOH")

    # des 对齐须在 cx 家族搬运之后,保证回折链贴在移动后的天然链上
    cmd.align("desp and name C4'", "nat and name C4'", cycles=8)

    nat_order = residues_in_order("nat")
    des_order = residues_in_order("desp")
    iface_sel = "byres (nat within 5.0 of lig)"
    iface_resn = set(residues_in_order(iface_sel))
    iface_pos = [i for i, r in enumerate(nat_order) if r in iface_resn]  # nat 0 基位置
    n_iface = len(iface_pos)
    if len(nat_order) == len(des_order):
        m = list(range(len(des_order)))
    else:
        m = nw_map(residue_letters("nat"), residue_letters("desp"))
    iface_idx = [m[p] + 1 for p in iface_pos
                 if m[p] is not None]                      # des 1 基 resi
    if iface_idx:
        idx_str = "+".join(str(i) for i in iface_idx)
        cmd.select("des_if", f"desp and resi {idx_str}")
        src = "direct" if len(nat_order) == len(des_order) else "NW"
        print(f"[{stem}] nat={len(nat_order)} des={len(des_order)} "
              f"iface={n_iface}->{len(iface_idx)} ({src}) idx={iface_idx[:12]}")
    else:
        cmd.select("des_if", "none")
        print(f"[{stem}] WARN iface mapping failed nat={len(nat_order)} des={len(des_order)}")

    cmd.hide("everything", "all")
    cmd.show("cartoon", "nat")
    cmd.color("natv", "nat")
    cmd.set("cartoon_transparency", 0.45, "nat")
    cmd.show("cartoon", "desp")
    cmd.color("desp", "desp")
    if iface_idx:
        cmd.color("ifblue", "des_if")
    if len(cmd.get_model("lig").atom) > 0:
        cmd.show("sticks", "lig")
        cmd.color("ligor", "lig")
        cmd.set("stick_radius", 0.28, "lig")
    else:
        print(f"[{stem}] WARN no ligand atoms selected ({lig_resn})")

    if ref_stem and ref_stem in _ref_views:
        cmd.set_view(_ref_views[ref_stem])
    else:
        cmd.orient("nat")
        cmd.zoom("nat or lig", 4)
        if stem in FAMILY_REF.values():
            _ref_views[stem] = cmd.get_view()
    cmd.png(os.path.join(OUT, f"{stem}.png"), width=1200, height=800,
            ray=True, dpi=450)
    print(f"[{stem}] rendered")

print("ALL DONE")
