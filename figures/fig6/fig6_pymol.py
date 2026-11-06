# -*- coding: utf-8 -*-
"""Fig 6b 结构渲染(PyMOL 出版级)。

用本项目 rna_if 环境的 open-source PyMOL 无头运行（无水印）:
    /root/miniconda3/envs/rna_if/bin/pymol -cq fig6_pymol.py

产出(透明底、光线追踪,图内不含文字 —— 配体名/残基号由 fig6_b 的
matplotlib 标注在渲染图上下给出):
  out/fig6_pymol_overview.png  8HZD 全景:链 A cartoon 灰绿 + 界面残基(<5 Å 配体)
                               雾蓝段 + 链 B 淡灰参照 + N618(O2I)橙色 stick
  out/fig6_pymol_pocket.png    口袋放大(<8 Å,与全景同取向):界面蓝/近端灰 +
                               配体 stick

注意:8HZD 真正结合的配体是荧光染料 N618(残基代码 O2I);GDP 只是链 B 的
5′-GMP 共价帽子,不作配体展示(旧版 fig6.py 误用)。
画幅 3000×1500(横幅,比例 2.0)= panel b 槽位 1.562×~0.80 in(2026-10-23
起内容为横条状 RNA 链/口袋,横幅取景使内容满框、与画廊列心对齐;
渲染后需再跑 fig6_trim.py 裁透明边),拼图不变形。
"""
import os

from pymol import cmd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
OUT = os.path.join(ROOT, "figures", "out")

# fig_style.py 语义色板(12 色板) → PyMOL
cmd.set_color("natv", [0xA3 / 255, 0xB1 / 255, 0x8A / 255])    # 天然结构灰绿
cmd.set_color("ifblue", [0x6B / 255, 0x7B / 255, 0x8D / 255])  # 界面残基石板蓝
cmd.set_color("pale", [0xB7 / 255, 0xB0 / 255, 0xA2 / 255])    # 非界面淡暖灰
cmd.set_color("ligor", [0xC0 / 255, 0x8B / 255, 0x7C / 255])   # 配体恒陶土橙
cmd.set_color("inkg", [0, 0, 0])                               # 标签墨色(黑)

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
# 平光与画廊一致(2026-10-23):关高光/雾化,提环境光,链 B 0.60→0.45
cmd.set("specular", 0)
cmd.set("depth_cue", 0)
cmd.set("ambient", 0.55)
cmd.set("direct", 0.40)
cmd.set("orthoscopic", "on")

cmd.hide("everything", "complex")
cmd.show("cartoon", "chA")
cmd.color("natv", "chA")
cmd.color("ifblue", "ifaceA")           # 界面残基卡通段直接染雾蓝
cmd.show("cartoon", "chB")
cmd.color("pale", "chB")
cmd.set("cartoon_transparency", 0.45, "chB")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")

# ---------- 全景视角 ----------
# 无头模式下取景比例由 png 的 width/height 决定(viewport 不生效);高分辨率
# 渲染 + fig6_trim.py 裁透明边后仍 ≥600 ppi
cmd.viewport(3000, 1500)
cmd.orient("chA")
cmd.rotate("x", -8)                      # 稍微俯视,口袋朝向读者
cmd.rotate("y", 20)
# 链 B 只显示配体近旁残基(远处链段会在视口边缘被平切)
cmd.select("chBshow", "chB byres (chB within 14.0 of lig)")
cmd.hide("cartoon", "chB")
cmd.show("cartoon", "chBshow")
cmd.color("pale", "chBshow")
cmd.set("cartoon_transparency", 0.45, "chBshow")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")
# 以"实际显示内容"取景(而非配体+固定缓冲),保证所有可见残基完整入画
cmd.center("chA or chBshow or lig")
cmd.zoom("chA or chBshow or lig", 12)

# 2026-10-23:图内不再嵌任何文字(旧 "N618 ×2" 标签与口袋残基号在 1.1 in
# 显示宽下相互压叠、不可读)——配体名与界面残基号改由 fig6_b 在渲染图
# 上下的 matplotlib 标注给出(黑色、7 pt、不与结构重叠)

cmd.viewport(3000, 1500)                 # 与取景一致
cmd.png(os.path.join(OUT, "fig6_pymol_overview.png"),
        width=3000, height=1500, ray=True)

# ---------- 口袋放大(同取向,仅换视野) ----------
VIEW = cmd.get_view()
cmd.hide("everything", "complex")
cmd.show("cartoon", "pocketA")
cmd.color("ifblue", "ifaceA")
cmd.color("pale", "pocketA and not ifaceA")
cmd.show("sticks", "lig")
cmd.color("ligor", "lig")
cmd.set("cartoon_tube_radius", 0.55, "pocketA")
cmd.set_view(VIEW)
cmd.viewport(3000, 1500)                  # 取景前定画幅(复用全景视角)
cmd.center("pocketA or lig")
cmd.zoom("pocketA or lig", 6)

cmd.viewport(3000, 1500)
cmd.png(os.path.join(OUT, "fig6_pymol_pocket.png"),
        width=3000, height=1500, ray=True)
print("pymol renders done")
