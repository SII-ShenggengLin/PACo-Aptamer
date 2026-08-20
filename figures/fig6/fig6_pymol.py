# -*- coding: utf-8 -*-
"""Fig 6a 结构渲染(PyMOL 出版级)。

用 egarps_env 的 PyMOL 无头运行:
    /root/miniconda3/envs/egarps_env/bin/pymol -cq fig6_pymol.py

产出(透明底、光线追踪):
  out/fig6_pymol_overview.png  8HZD 全景:链 A cartoon 灰绿 + 界面残基(<5 Å 配体)
                               雾蓝段 + 链 B 淡灰参照 + N618(O2I)橙色 stick
  out/fig6_pymol_pocket.png    口袋放大(<8 Å,与全景同取向):界面蓝/近端灰 +
                               残基编号 + 配体 stick

注意:8HZD 真正结合的配体是荧光染料 N618(残基代码 O2I);GDP 只是链 B 的
5′-GMP 共价帽子,不作配体展示(旧版 fig6.py 误用)。
画幅比例与 fig6_common.FIG6_LAYOUT 中 a1/a2 槽位一致,拼图不变形。
"""
import os

from pymol import cmd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
OUT = os.path.join(ROOT, "figures", "out")

# fig_style.py 语义色板 → PyMOL
cmd.set_color("natv", [0x7F / 255, 0xA8 / 255, 0x8B / 255])    # 天然结构灰绿
cmd.set_color("ifblue", [0x6C / 255, 0x9B / 255, 0xD1 / 255])  # 界面/设计雾蓝
cmd.set_color("pale", [0xC8 / 255, 0xCF / 255, 0xC5 / 255])    # 非界面淡灰
cmd.set_color("ligor", [0xE8 / 255, 0xA0 / 255, 0x88 / 255])   # 配体恒橙
cmd.set_color("inkg", [0x54 / 255, 0x54 / 255, 0x50 / 255])    # 标签墨灰

LIG = "resn O2I"

cmd.load(os.path.join(ROOT, "data", "rcsb_pdb", "8HZD.pdb"), "complex")
cmd.remove("resn HOH")
cmd.select("chA", "chain A")
cmd.select("chB", "chain B")
cmd.select("lig", LIG)
cmd.select("ifaceA", "chA byres (chA within 5.0 of lig)")
cmd.select("pocketA", "chA byres (chA within 8.0 of lig)")
cmd.select("pocketB", "chB byres (chB within 8.0 of lig)")

# ---------- 全局渲染设置 ----------
cmd.bg_color("white")
cmd.set("ray_opaque_background", "off")
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.set("orthoscopic", "off")
cmd.set("cartoon_sampling", 14)
cmd.set("cartoon_smooth_cycles", 2)
cmd.set("cartoon_tube_radius", 0.42)
cmd.set("cartoon_tube_cap", 2)
cmd.set("stick_radius", 0.16)
cmd.set("transparency_mode", 2)
cmd.set("depth_cue", "on")

cmd.hide("everything", "complex")
cmd.show("cartoon", "chA")
cmd.color("natv", "chA")
cmd.color("ifblue", "ifaceA")           # 界面残基卡通段直接染雾蓝
cmd.show("cartoon", "chB")
cmd.color("pale", "chB")
cmd.set("cartoon_transparency", 0.60, "chB")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")

# ---------- 全景视角 ----------
cmd.orient("chA")
cmd.rotate("x", -8)                      # 稍微俯视,口袋朝向读者
cmd.rotate("y", 20)
cmd.zoom("chA or lig", 4)

# N618 标签(伪原子挂在两份配体质心外上方,避开卡通)
cx, cy, cz = cmd.centerofmass("lig")
cmd.pseudoatom("liglab", label="N618 ×2", pos=[cx + 12, cy + 10, cz])
cmd.set("label_color", "inkg", "liglab")
cmd.set("label_size", 16, "liglab")

cmd.viewport(1160, 1560)                 # a1 槽 41.2×55.4mm,比例 0.744
cmd.png(os.path.join(OUT, "fig6_pymol_overview.png"),
        width=1160, height=1560, ray=True)

# ---------- 口袋放大(同取向,仅换视野) ----------
VIEW = cmd.get_view()
cmd.hide("everything", "complex")
cmd.show("cartoon", "pocketA")
cmd.color("ifblue", "ifaceA")
cmd.color("pale", "pocketA and not ifaceA")
cmd.show("cartoon", "pocketB")
cmd.color("pale", "pocketB")
cmd.set("cartoon_transparency", 0.75, "pocketB")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")
cmd.set("cartoon_tube_radius", 0.55, "pocketA")
cmd.label("pocketA and name C1'", "resi")
cmd.set("label_color", "inkg", "pocketA")
cmd.set("label_size", 13, "pocketA")
cmd.set_view(VIEW)
cmd.zoom("lig", 9.5)

cmd.viewport(840, 1582)                  # a2 槽 29.4×55.4mm,比例 0.531
cmd.png(os.path.join(OUT, "fig6_pymol_pocket.png"),
        width=840, height=1582, ray=True)
print("pymol renders done")
