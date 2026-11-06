# -*- coding: utf-8 -*-
"""Fig 1 真实结构素材渲染(PyMOL 无头,出版级)。

示例链:1U8D 链 A(鸟嘌呤核糖开关,配体 HPA;RNAfold 预测为三向连接,
素材间保持同一示例链,见图题)。

用法:/root/miniconda3/envs/rna_if/bin/pymol -cq figures/fig1/fig1_pymol.py
产出(figures/fig1/assets/,透明底、光线追踪):
  fig1_bb.png     面板 a 输入:1U8D 链 A 全长骨架(亮石板蓝 cartoon)
  fig1_nt.png     面板 a 输入:1U8D 链 A 第 13 位 G(resi 27)的球棍特写,
                  PACO9 九原子(P O5' C5' C4' C3' C2' C1' N1 C2,与数据管线
                  的原子名一致)深蓝、其余原子浅灰
  fig1_ss 由 fig1_assets_data.py 产出(RNAplot 布局,非本脚本)
  fig1_pocket.png 面板 a 输入:口袋特写 —— 全链浅蓝卡通作背景脉络,
                  口袋残基(≤4.5 Å)饱和蓝球棍围出结合腔,配体 HPA
                  陶土橙球棍居中(v1.43 起配体与 RNA 同表征:实心球团
                  与周围骨架画法不属于一种类型,改为球棍后与口袋残基、
                  图6 的配体画法全文一致)
  fig1_graph.png  面板 a GVP 框:1U8D 链 A 真实几何图 —— C4' 节点球,
                  主链边(近黑实线)、kNN 边(k=15,暖灰虚线)、
                  二级结构配对边(陶土虚线,RNAfold 预测),
                  边表来自 fig1_assets_data.py 的 fig1_1u8d.json

出版级渲染要点(v1.28 升级,替代旧的平光单灯配方;v1.42 口袋改表征):
  * 三点柔光(light_count 3)+ 抬高环境光、压低直射与镜面,白底上
    呈现柔和立体感而非旧版的平淡灰调;
  * surface_quality 2 / sphere_quality 2 / cartoon_sampling 21,
    表面与球体平滑无多边形感;
  * 所有素材按 2× 超采样渲染(尺寸翻倍、纵横比不变),落到图内
    缩小后边缘锐利;
  * 对比度整体加深:骨架/节点/配体色在主题色系内各深一档,配体
    球更饱和,几何图 kNN 边加深到可辨;
  * v1.42 口袋素材根治糊(v1.29/v1.41 两轮只调光照与 carve 半径,
    均未解决):carve 分子表面是平滑连续壳,内部沟壑的明暗变化在
    面板显示高度(~14 mm)下低于视觉分辨极限,任何光照/透明度配置
    都读作无结构的蓝色斑块 —— 表征本身的限制而非渲染分辨率问题
    (全分辨率素材始终锐利)。改为口袋特写球棍表征,与上方骨架卡通
    同级可读;v1.43 配体由实心球改球棍,与 RNA 表征类型统一。

配色跟随 Fig 1 明亮主题(fig1_common.py 为唯一事实来源):
RNA 亮蓝系,配体陶土橙点缀。取景一律以"实际显示的选区"为准
(隐藏原子不参与 zoom),避免空旷边距。
"""
import json
import os

from pymol import cmd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
OUT = os.path.join(ROOT, "figures", "fig1", "assets")
os.makedirs(OUT, exist_ok=True)

cmd.set_color("rnab", [0x6F / 255, 0x92 / 255, 0xC0 / 255])    # 亮石板蓝(骨架,v1.29 再深一档,小尺寸下不显淡)
cmd.set_color("rnap", [0x4E / 255, 0x86 / 255, 0xC6 / 255])    # 饱和亮蓝(PACO9 高亮)
cmd.set_color("cartc", [0xB9 / 255, 0xD0 / 255, 0xE3 / 255])   # 浅主题蓝(口袋特写背景卡通;比球棍浅一档,拉开明度分层)
cmd.set_color("nodec", [0x9D / 255, 0xBF / 255, 0xDD / 255])   # 图节点
cmd.set_color("ligor", [0xA9 / 255, 0x63 / 255, 0x4B / 255])   # 配体深陶土橙
cmd.set_color("atomgrey", [0.88, 0.88, 0.88])                  # 非 PACO9 原子
cmd.set_color("knngray", [0xB5 / 255, 0xAF / 255, 0xA3 / 255]) # 暖灰 kNN 边
cmd.set_color("bbink", [0x1A / 255, 0x1A / 255, 0x1A / 255])   # 近黑主链边

# ---------- 全局出版级渲染设置 ----------
cmd.bg_color("white")
cmd.set("ray_opaque_background", "off")
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.set("orthoscopic", "on")
cmd.set("two_sided_lighting", "on")
cmd.set("transparency_mode", 2)
# 三点柔光:主光 + 补光 + 轮廓光,环境光抬高、直射与镜面压低
cmd.set("light_count", 3)
cmd.set("light", [-0.4, -0.4, -1.0])
cmd.set("light2", [-0.55, -0.7, 0.15])
cmd.set("light3", [0.3, -0.6, -0.2])
cmd.set("ambient", 0.34)
cmd.set("direct", 0.32)
cmd.set("reflect", 0.30)
cmd.set("reflect_power", 1.6)
cmd.set("specular", 0.32)
cmd.set("specular_intensity", 0.35)
cmd.set("spec_direct", 0.12)
cmd.set("spec_power", 42)
# 采样与平滑(出版级):表面/球体/卡通均高质量
cmd.set("surface_quality", 2)
cmd.set("sphere_quality", 2)
cmd.set("cartoon_sampling", 21)
cmd.set("cartoon_smooth_cycles", 3)
cmd.set("cartoon_tube_radius", 0.55)
cmd.set("cartoon_tube_cap", 2)
cmd.set("stick_radius", 0.16)
# 轻微景深雾增强立体(远处稍淡)
cmd.set("depth_cue", "on")
cmd.set("fog_start", 0.62)


def _reset():
    cmd.delete("all")


def _shot(name, sel, w, h, buffer=2.0, rot=("z", 0)):
    """以可见选区取景并渲染(2× 超采样尺寸由调用方给出)。

    rot 在 orient 之后施加一次旋转再重新取景,用于把结构转到更舒展的
    朝向(orient 只保证主轴对齐,不保证最美观)。
    """
    cmd.orient(sel)
    if rot[1]:
        cmd.rotate(rot[0], rot[1])
    cmd.center(sel)
    cmd.zoom(sel, buffer)
    cmd.png(os.path.join(OUT, name), width=w, height=h, dpi=300, ray=1)
    print("saved", name)


_LIGHT_BASE = {"ambient": 0.34, "direct": 0.32, "reflect": 0.30,
               "specular": 0.32, "specular_intensity": 0.35,
               "spec_power": 42}


def _light(**over):
    """按场景覆盖全局光照(默认 _LIGHT_BASE,其余场景复位)。"""
    s = dict(_LIGHT_BASE)
    s.update(over)
    for k, v in s.items():
        cmd.set(k, v)


# ---------- 1. 面板 a:RNA 骨架(1U8D 链 A 全长) ----------
_reset()
cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb"), "cplx")
cmd.remove("resn HOH")
cmd.select("chA", "chain A")
cmd.hide("everything")
cmd.show("cartoon", "chA")
cmd.color("rnab", "chA")
cmd.set("cartoon_tube_radius", 0.66, "chA")
# 朝向:三向连接近似共面,正对该平面观看时三茎分开最舒展(投影凸包
# 面积 × 平均最近邻距离最大,对平面内旋转不敏感)。以 C4' 主轴系
# SVD 对齐后不施加任何倾角,直接把坐标转到正面视角(transform_object
# 替代 orient,视角确定可复现)。
import numpy as np
c4 = []
for line in open(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb")):
    if line.startswith("ATOM") and line[21] == "A" and line[12:16].strip() == "C4'":
        c4.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
P = np.asarray(c4)
Pc = P - P.mean(0)
_, _, Vt = np.linalg.svd(Pc, full_matrices=False)
M = Vt.T                                  # 主轴对齐;x-y 即连接平面
m16 = [M[0, 0], M[0, 1], M[0, 2], 0.0,
       M[1, 0], M[1, 1], M[1, 2], 0.0,
       M[2, 0], M[2, 1], M[2, 2], 0.0,
       0.0, 0.0, 0.0, 1.0]
cmd.transform_object("cplx", m16)
cmd.center("chA")
cmd.zoom("chA", 1.6)
# 骨架 tube 需要更强的明暗雕刻:直射与镜面抬高一档,高光更收紧
_light(ambient=0.30, direct=0.50, reflect=0.22, specular=0.45,
       specular_intensity=0.55, spec_power=60)
cmd.png(os.path.join(OUT, "fig1_bb.png"), width=2400, height=1800,
        dpi=300, ray=1)
print("saved fig1_bb.png")
_light()
cmd.set("cartoon_tube_radius", 0.55)

# ---------- 2. 面板 a:配体口袋(口袋残基球棍 + 背景卡通 + 配体球) ----------
_reset()
cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb"), "cplx")
cmd.remove("resn HOH")
# 配体 HPA 以 HET 记录挂在链 A 上,chA 必须限定 polymer 原子,
# 否则配体参与取景选区并带入元素默认色
cmd.select("chA", "chain A and polymer")
cmd.select("lig", "resn HPA and chain A")
cmd.select("pocket", "byres (chA within 4.5 of lig)")
cmd.hide("everything")
# 取向(确定性,替代 orient):把"口袋壁质心 → 配体质心"方向对准
# 相机一侧(PyMOL 相机在 −z,目标 (0,0,−1)),配体朝前、口袋壁在
# 后侧展开;v1.41 烘焙的 −8°/+8° 倾角一并保留(transform_object
# 跨进程可复现,相机 cmd.rotate 不可复现)
import numpy as np
_pdb = open(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb"))
_pk, _lg = [], []
for _l in _pdb:
    if not (_l.startswith("ATOM") or _l.startswith("HETATM")) or _l[21] != "A":
        continue
    if _l[17:20].strip() == "HOH":
        continue
    _xyz = [float(_l[30:38]), float(_l[38:46]), float(_l[46:54])]
    (_lg if _l.startswith("HETATM") else _pk).append(_xyz)
_pk = np.asarray(_pk)
_lg = np.asarray(_lg)
_lig_c = _lg.mean(0)
_near = np.linalg.norm(_pk - _lig_c, axis=1) <= 4.5
_v = _lig_c - _pk[_near].mean(0)
_v /= np.linalg.norm(_v)
_z = np.array([0.0, 0.0, -1.0])
_ax = np.cross(_v, _z)
_s = np.linalg.norm(_ax)
_c = float(_v @ _z)
if _s < 1e-6:
    _R = np.eye(3) if _c > 0 else np.diag([1.0, -1.0, -1.0])
else:
    _ax = _ax / _s
    _K = np.array([[0, -_ax[2], _ax[1]], [_ax[2], 0, -_ax[0]],
                   [-_ax[1], _ax[0], 0]])
    _R = np.eye(3) + _s * _K + (1 - _c) * (_K @ _K)
# v1.41 视角微调烘焙进对象变换(相机 cmd.rotate 跨进程不可复现,弃用):
# 基础取向上再绕世界 x −8°、y +8°,与 carve 12 Å 联合闭合 carved 表面
# 在三向连接开口处的透白大洞(81 视角 × carve 网格搜索的零洞解之一,
# 全分辨率复检:最终图仅剩 ~0.6 kpx 针尖斑,缩入面板后不可辨)
_tx, _ty = np.radians(-8.0), np.radians(8.0)
_Rx = np.array([[1, 0, 0],
                [0, np.cos(_tx), -np.sin(_tx)],
                [0, np.sin(_tx), np.cos(_tx)]])
_Ry = np.array([[np.cos(_ty), 0, np.sin(_ty)],
                [0, 1, 0],
                [-np.sin(_ty), 0, np.cos(_ty)]])
_Rt = _Ry @ _Rx @ _R
_t = -_Rt @ _lig_c
_m16 = [_Rt[0, 0], _Rt[0, 1], _Rt[0, 2], _t[0],
        _Rt[1, 0], _Rt[1, 1], _Rt[1, 2], _t[1],
        _Rt[2, 0], _Rt[2, 1], _Rt[2, 2], _t[2],
        0.0, 0.0, 0.0, 1.0]
cmd.transform_object("cplx", _m16)
# v1.42 根治口袋糊:carve 分子表面(v1.28–v1.41 的表征)是平滑连续
# 壳,缩到面板显示高度(~14 mm)后内部沟壑的明暗低于视觉分辨极限,
# 读作无结构的蓝色斑块 —— 表征本身的限制,而非渲染分辨率问题(素材
# 全分辨率一直锐利,v1.29 不透明化与 v1.41 补洞都没触及根源)。改为
# 口袋特写的常规三级表征:全链浅蓝卡通作背景脉络(三茎汇入连接处,
# 与上方骨架卡通同表征、天然小尺寸可读)、口袋残基(≤4.5 Å)饱和蓝
# 球棍围出结合腔、配体 HPA 陶土橙居中;浅蓝卡通 → 蓝球棍 → 橙配体的
# 明度+色相阶梯保证面板尺寸下各自可辨。取景口径 ~23 Å(zoom 缓冲 9):
# 再近则口袋残基碱基粘连成板、卡通只剩边角碎段(首渲 5.5 的教训)。
# v1.43 配体表征统一:实心球团与周围的骨架/球棍画法不属于一种类型
# (用户反馈"配体就是一团"),改为与口袋残基、图6 配体一致的球棍,
# 半径略粗于 RNA 残基(0.26 vs 0.20)保住视觉焦点地位。单次光线追踪
# 出图,不再分层合成、无需针孔修补。
cmd.show("cartoon", "chA")
cmd.color("cartc", "chA")
cmd.set("cartoon_tube_radius", 0.48, "chA")
cmd.show("sticks", "pocket")
cmd.color("rnap", "pocket")
cmd.set("stick_radius", 0.20, "pocket")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")
cmd.set("stick_radius", 0.26, "lig")
_light(ambient=0.24, direct=0.58, reflect=0.18, specular=0.42,
       specular_intensity=0.50, spec_power=60)
cmd.center("lig")
cmd.zoom("lig", 9.0)
cmd.png(os.path.join(OUT, "fig1_pocket.png"), width=2000, height=1720,
        dpi=300, ray=1)
print("saved fig1_pocket.png")
_light()

# ---------- 3. 面板 a:单核苷酸 9 原子球棍(1U8D 链 A resi 27 的 G) ----------
_reset()
cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb"), "cplx")
cmd.select("nt", "chain A and resi 27")
cmd.hide("everything")
cmd.show("sticks", "nt")
cmd.show("spheres", "nt")
cmd.set("stick_radius", 0.15, "nt")
cmd.color("atomgrey", "nt")
cmd.set("sphere_scale", 0.26, "nt")
# PACO9(与数据管线原子名一致,字面 N1):P O5' C5' C4' C3' C2' C1' N1 C2
paco9 = "name P or name O5' or name C5' or name C4' or name C3' or " \
        "name C2' or name C1' or name N1 or name C2"
cmd.color("rnap", f"nt and ({paco9})")
cmd.set("sphere_scale", 0.40, f"nt and ({paco9})")
cmd.orient(f"nt and ({paco9})")
cmd.rotate("y", 30)
cmd.center("nt")
cmd.zoom("nt", 2.3)
cmd.png(os.path.join(OUT, "fig1_nt.png"), width=1800, height=1520,
        dpi=300, ray=1)
print("saved fig1_nt.png")

# ---------- 4. 面板 a GVP 框:真实几何图(C4' 节点 + 三类边) ----------
_reset()
with open(os.path.join(OUT, "fig1_1u8d.json")) as f:
    g = json.load(f)
resis = g["resis"]
cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb"), "cplx")
cmd.select("c4", "chain A and polymer and name C4'")
cmd.hide("everything")
cmd.show("spheres", "c4")
cmd.set("sphere_scale", 0.85, "c4")
cmd.color("nodec", "c4")
bb = [(i, i + 1) for i in range(len(resis) - 1)]
for obj, pairs in (("e_bb", bb), ("e_knn", g["pairs_knn"]),
                   ("e_pair", g["pairs_ss"])):
    for i, j in pairs:
        cmd.distance(obj, f"c4 and resi {resis[i]}",
                     f"c4 and resi {resis[j]}")
cmd.set("dash_color", "bbink", "e_bb")        # 主链:近黑实线
cmd.set("dash_gap", 0.0, "e_bb")
cmd.set("dash_radius", 0.20, "e_bb")
cmd.set("dash_color", "knngray", "e_knn")     # kNN:暖灰细虚线
cmd.set("dash_radius", 0.04, "e_knn")
cmd.set("dash_gap", 0.40, "e_knn")
cmd.set("dash_color", "ligor", "e_pair")      # 配对:陶土虚线
cmd.set("dash_radius", 0.12, "e_pair")
cmd.set("dash_gap", 0.30, "e_pair")
cmd.hide("labels")
_shot("fig1_graph.png", "c4", 2200, 1800, buffer=3.0)
