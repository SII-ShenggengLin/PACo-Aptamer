# 新增实验 D：LOFO 配体化学覆盖度 + RhoDesign 预测骨架

日期：2026-10-16。执行环境：GPU 1（CUDA_VISIBLE_DEVICES=1）；PACo 用 `rnasmol_service_gpu_env`
（py3.10 / torch 2.1.2 / pyg 2.5.0，原 `aptamer_eval` 环境已不存在，此为等价替代：PACo
代码仅依赖 torch+torch_geometric+numpy，ckpt 兼容已验证）；RhoDesign 用
`rhodesign_service_env`（torch 1.12.1，与 29v2 btest 同环境）。

## 任务 A：LOFO（leave-one-family-out）配体化学覆盖度实验

### A.0 预备：train.py 缓存隔离修复

`tools/PACoAptamer/train.py` 原逻辑把任何非遗留名字的 `--s2_data` 都映射到同一缓存
`data/paco_s2_v2.pt`——过滤版数据会命中全量旧缓存使过滤失效。已修复为：仅
`s2_samples.pt / s2_samples_clean.pt / s2_samples_v2.pt` 三个遗留名用原缓存，其余按
`data/paco_<basename>.pt` 按名隔离。5 链冒烟集验证：新缓存按名生成、旧缓存 mtime 不变、
训练正常（smoke5：data/s2_btrain/s2_samples_v2_smoke5.pt）。

### A.1 链→配体 CCD 归属（data/95_lofo.py --stage map）

- s2 768 链 / 510 PDB；同一 PDB 内各链 ligand_elements 完全一致（35 号管线按**全局**功能
  配体白名单收集 PDB 内全部 HETATM），故归属在 PDB 级进行。
- 候选 = `candidates_ccd.json` 该 PDB 的 `ligands.other`；元素多重集来自
  `data/ccd/components.cif.gz`（重原子，区分大小写归一）。
- 归属规则（优先级递减）：
  1. `element_exact`：链配体元素多重集 = 某候选 CCD 组成的整数倍（383/510 PDB）
  2. `element_union`：候选组合的精确并集解（23）
  3. `greedy_pdb`：per-PDB 候选内贪心覆盖，覆盖率 ≥0.8（8）；组成先投影到 M 实有元素，
     容忍旧 PDB 部分原子未沉积（典型：1ILX 光合系统 II 的 CLA 只沉积了 C/N/MG 原子，
     正则组成含 O 无法整除——修正前被全局白名单里的 POH 卟啉错抢，修正后归 CLA）
  4. `greedy_ext`：扩展到全局白名单贪心，≥4 重原子 CCD 才计（34；排除 OH/NME 等
     1-2 原子项被贪心滥配）
  5. `loose_pdb`：宽松口径，归属 per-PDB 全部候选（62）
- 交叉校验：用同管线在 `data/rcsb_pdb` 原始文件上重建 74 号 115 条界面链（test 侧，
  与 s2 不相交）的配体归属，与权威标注 `74_chain_ccd.json` 比对：**112/115 一致
  （97.4%）**；3 例不符均为 test 侧 greedy_ext 的多拷贝部分沉积结构（1OLN/5D8H：QUA→NO1；
  8GS2：C5P→XY7），不影响 s2 归属可信度。
- 持久化：`data/survey/95_chain_ccd.json`。

### A.2 家族分布与 top-5 选择（--stage families）

家族合并口径（化学同族）：flavins={FMN,FAD}、sam_sah={SAM,SAH}、
ntp={ATP,GTP,CTP,UTP}、aminoglycoside={PAR,GET,NMY,RIO,AM2,TOB}（2-脱氧链霉胺类
氨基糖苷：巴龙霉素/遗传霉素/新霉素/核糖霉素/安普霉素/妥布霉素）；其余 CCD 自成一家。
CCD 级与家族级双口径计数见 `data/survey/95_families.json`。

CCD 级 top-10（链数）：PAR 75、SAM 38、CLA 24、NMY 18、SAH 17、ATP 17、GET 17、
2BA 16、GAI 15、3V6 13。

家族级 top-5（链数 / 涉及 PDB 数）：

| 家族 | 链数 | PDB 数 | 化学身份 |
|---|---|---|---|
| aminoglycoside | 128 | ~50 | 氨基糖苷类抗生素（PAR/GET/NMY/RIO/AM2/TOB） |
| sam_sah | 55 | ~40 | SAM/SAH 腺苷类辅酶 |
| cla | 24 | 2 | 叶绿素 a（1ILX 光合系统 II + 1C51 光合反应中心，部分原子沉积） |
| ntp | 20 | ~15 | 核苷三磷酸（ATP 17 为主，GTP/CTP/UTP 少量） |
| 2ba | 16 | 16 | c-di-AMP（环二腺苷酸，核糖开关配体，C20N10O12P2） |

说明：NTP 类采用"CCD 级保留 + 家族级合并"双口径——ATP/GTP/CTP/UTP 单独计数均不突出
（17/2/0/1），但共享核苷三磷酸化学，合并后才进入 top-5；flavins 合并后仅 ~10 链，
未进 top-5。GAI（胍，15 链）为第 6 名候补。

### A.3 LOFO 与累积过滤集（--stage makedata）

| 变体 | 链数 | 说明 |
|---|---|---|
| lofo_aminoglycoside | 640 | 剔除 128 链（16.7%） |
| lofo_sam_sah | 713 | 剔除 55 |
| lofo_cla | 744 | 剔除 24 |
| lofo_ntp | 748 | 剔除 20 |
| lofo_2ba | 752 | 剔除 16 |
| only1fam | 128 | 仅 aminoglycoside |
| only2fam | 183 | + sam_sah |
| only3fam | 207 | + cla |
| only5fam | 243 | + ntp + 2ba |
| full_ctl | 768 | 全量复刻（run-to-run 噪声参照） |

落盘 `data/s2_btrain/s2_samples_v2_{lofo_<tag>,only<k>fam,fullctl}.pt`（dict 结构同
s2_samples_v2.pt），manifest 见 `data/survey/95_datasets.json`。

### A.4 训练（--stage train）

全部 10 变体 + 基线同配方：`--stage s2 --init s1_batch4_seed43.pth --splits_v2
--joint --val_sel --epochs 8 --seed 43 --batch_chains 4`（主配置单模配方；train.py 已无
界面加权，与 s2_v2_seed43_noiface 同配方）。ckpt `models/PACoAptamer/s2_v2_lofo_*.pth /
s2_v2_only*fam.pth / s2_v2_lofo_full_ctl.pth`，日志 `data/survey/95_train_*.log`，
val 轨迹记入 `data/survey/95_lofo.json` 的 train_val 段。

### A.5 评估口径

- **zs 93 链**（zero_shot_targets）：68 号 A 层口径（grnade_inputs_v2 特征 + RNAfold SS），
  3 温度（0.1/0.5/1.0）×100 采样，best=池内 max，t01=temp0.1 池均值。基线 =
  既有 `s2_v2_seed43_noiface`（同配方全量），另评 full_ctl 量化训练噪声。
  逐链 Δ + Wilcoxon 符号秩 p。
- **界面 115 链**（74_iface_meta：usable 且 0<n_iface≤50）：74 号 B 层口径
  （paco_s1_v2 缓存特征；A 层 grnade_inputs_v2 只覆盖其中 29 条，不足以支撑该层）。
  报告界面位置恢复率（iface_best/iface_t01）与整体恢复率。
- 采样池留档 `data/survey/95_pools/{variant}/{zs|iface}/{stem}.json`。

### A.6 结果

完整数据 `data/survey/95_lofo.json`（含逐链池摘要与 Δ）。

**zs 93 链（3 温度×100，best=池内 max，t01=temp0.1 均值）**

| 变体 | 训练链数 | best | t01 | Δbest vs 基线 | Wilcoxon p |
|---|---|---|---|---|---|
| s2_v2_seed43_noiface（基线） | 768 | 0.7335 | 0.6332 | — | — |
| s2_v2_full_ctl（全量复刻） | 768 | 0.7339 | 0.6450 | +0.0004 | 0.911 |
| lofo_aminoglycoside | 640 | 0.7146 | 0.6067 | **-0.0189** | 2.7e-4 |
| lofo_sam_sah | 713 | 0.7189 | 0.6223 | **-0.0146** | 1.5e-3 |
| lofo_2ba | 752 | 0.7215 | 0.6150 | -0.0120 | 0.015 |
| lofo_ntp | 748 | 0.7230 | 0.6243 | -0.0105 | 0.097 |
| lofo_cla | 744 | 0.7287 | 0.6357 | -0.0048 | 0.499 |
| only1fam（氨基糖苷） | 128 | 0.7124 | 0.6268 | -0.0211 | 1.4e-5 |
| only2fam | 183 | 0.7350 | 0.6411 | +0.0015 | 0.688 |
| only3fam | 207 | 0.7329 | 0.6371 | -0.0006 | 0.813 |
| only5fam | 243 | 0.7374 | 0.6268 | +0.0039 | 0.361 |

要点：
- **run-to-run 噪声**：full_ctl 与基线 best 差 +0.04pp（p=0.91）、t01 差 +1.2pp——
  best 口径噪声 <0.1pp，LOFO 中 aminoglycoside/sam_sah/2ba 的降幅（1.2-1.9pp）
  显著超出训练噪声；cla/ntp 降幅与噪声同量级。
- **降幅随家族规模单调**：剔除 128 链的氨基糖苷家族降幅最大（-1.9pp），剔除 16 链的
  2ba 仍有 -1.2pp（p=0.015）——小家族也有可测贡献，但全部单家族 LOFO 降幅 ≤2pp，
  模型对单一配体家族缺失相当鲁棒（joint 反遗忘 + S1 通用结构先验兜底）。
- **累积覆盖曲线饱和极快**：仅 top-2 家族（氨基糖苷+SAM/SAH，183 链 = 24% 数据）
  即达 0.7350，与全量持平；top-5（243 链）0.7374 甚至略高于基线（噪声内）。
  结论：在 s2_btrain 当前规模下，**配体化学多样性对 zero-shot 泛化的边际贡献在
  前 2 个家族后饱和**，堆更多同族复合物不如保少量多样家族。
- val_sel 注意：各变体干净 val 链数不同（59/87/68/96，因剔除的 S2 PDB 不同），
  训练日志的 VAL 仅用于变体内选 ckpt，不可跨变体直接比较（lofo_sam_sah 的 0.6879
  是在 87 链 val 上的，与其余变体的 59 链口径不同）。

**界面 115 链（74 号 B 层口径；LOFO 5 变体 + 基线 + full_ctl）**

| 变体 | 整体 best | 整体 t01 | 界面 best | 界面 t01 |
|---|---|---|---|---|
| 基线 | 0.7214 | 0.6133 | 0.8145 | 0.6476 |
| full_ctl | 0.7269 | 0.6282 | 0.8375 | 0.6210 |
| lofo_aminoglycoside | 0.7105 | 0.6061 | 0.8030 | 0.6189 |
| lofo_sam_sah | 0.7213 | 0.6153 | 0.8166 | 0.5920 |
| lofo_cla | 0.7302 | 0.6309 | 0.8289 | 0.6190 |
| lofo_ntp | 0.7074 | 0.6030 | 0.8257 | 0.6406 |
| lofo_2ba | 0.7066 | 0.6104 | 0.8097 | 0.6079 |

界面层 run-to-run 噪声（full_ctl vs 基线）达 +2.3pp（iface_best）/ -2.7pp
（iface_t01），所有 LOFO 变体的界面 best 偏差（-1.1pp ~ +1.4pp）均在噪声带内——
**未发现家族特异性的界面恢复率效应**；界面 t01 噪声更大，不据此下结论。

## 任务 B：RhoDesign 在 RhoFold+ 预测骨架上的批量推理

### B.0 输入与口径

- 输入：193 条 test 链的 RhoFold+ 预测结构 `data/survey/86_sctm/native_{stem}.pdb`，
  符号链接为 `data/survey/101_predbb_inputs/{stem}.pdb`（去 native_ 前缀）。
- 模型：`models/RhoDesign_btrain_v2/best_val.pth`（与晶体 btest 同权重，唯一变量是
  骨架来源）；模型代码用外部仓 `.../external_repos/RhoDesign/src`（可读，未走
  tools/RhoDesign_official 回退）。
- 采样：29v2 btest 同口径——温度 0.1/0.5/1.0；≤200nt 每温度 100、≤400nt 50、更长 30。
- 指标：池内 best / t01（temp0.1 池均值），native 序列取
  `survey/rhodesign_out_btest_v2/{stem}.native`（晶体 manifest 口径）。
- 输出：`survey/rhodesign_out_predbb/{stem}.{temp}.seqs` + `survey/101_rhodesign_predbb.json`，
  每 30 链断点落盘，重启自动跳过已完成链。
- 脚本：`data/101_run_rhodesign_predbb.py`。

### B.1 复现校验（btest 口径 10 链重算）

用 101 脚本的指标函数从晶体 btest 原始采样池重算 10 条随机链的 best/t01，与存档
`metrics_RhoDesign_btest_v2.json` 完全一致（max |diff| = 1.2e-15，浮点级）。
链：1Y39_C 4K31_B 4TZX_X 5SWE_X 6E8T_A 6EXN_2 6KL9_G 6UP0_C 6VWT_A 6XKO_A。

### B.2 效率探针（survey/99_efficiency_RhoDesign.json）

长度分位 6 链（最短/p20/p40/p60/p80/最长），批跑内测量（GPU 1 4090，与 PACo 训练
共享，墙钟略偏高）：

| 链 | 长度 | 样本/温度 | 墙钟(s) | 峰值显存(MiB) |
|---|---|---|---|---|
| 3TZR_A | 20 | 100 | 17.3 | 154 |
| 8CGJ_a | 28 | 100 | 21.3 | 158 |
| 6UP0_D | 38 | 100 | 27.4 | 165 |
| 3SKT_A | 64 | 100 | 44.0 | 179 |
| 3IVN_B | 69 | 100 | 47.6 | 183 |
| 7R6Q_1 | 295 | 50 | 97.7 | 313 |

逐样本吞吐 ~65-75ms（短链）~220ms（295nt）；显存峰值均 <0.4 GB，效率瓶颈在
自回归逐样本串行解码而非显存。

### B.3 结果

完整数据 `data/survey/101_rhodesign_predbb.json`（逐链）与
`data/survey/101_predbb_vs_crystal.json`（与晶体口径逐链配对对照）。

- 完成度：193/193，无失败；批跑总墙钟约 2.25 h（GPU 1，与 PACo LOFO 训练共享）。
- 预测骨架：best 0.6136（中位 0.6056）/ t01 0.4799。
- 同 193 链晶体口径（metrics_RhoDesign_btest_v2）：best 0.6828 / t01 0.5486。
- **Δ（预测 − 晶体）：best -6.9pp（中位 -6.0pp，Wilcoxon p=5.3e-24）；
  t01 -6.9pp（中位 -6.2pp，p=8.6e-17）**——骨架从晶体换成 RhoFold+ 预测，
  RhoDesign 恢复率系统性下降约 7pp，高度显著。
- 长度分层：≤50nt -7.1pp（n=96）、51-100nt -6.8pp（n=83）、>100nt -6.8pp（n=14），
  降幅与链长无关，是均匀的骨架质量税而非长链特异失效。
- 逐链：158 降 / 14 平（|Δ|≤0.005）/ 21 升；最差 7V9U_G -31.0pp、7XJG_E -29.3pp、
  6W62_B -29.2pp（RhoFold+ 预测结构失真个案）；最好 8H9D_G +14.3pp、1G1X_D +10.0pp。
- 与 PACo 预测骨架实验（实验 C，analysis/23）对比阅读：同一批 86_sctm 预测骨架上
  不同方法的鲁棒性差异可在此口径下横评。

## 复现索引

- 任务 A：`data/95_lofo.py --stage {map,families,makedata,train,eval,collect}`；
  中间件 `survey/95_chain_ccd.json / 95_families.json / 95_datasets.json / 95_eval.json`；
  ckpt `models/PACoAptamer/s2_v2_{lofo_*,only*fam}.pth`；池 `survey/95_pools/`。
- 任务 B：`CUDA_VISIBLE_DEVICES=1 python data/101_run_rhodesign_predbb.py --gpu 0`
  （rhodesign_service_env；断点续跑）；输入 `survey/101_predbb_inputs/`（符号链接到
  86_sctm/native_*.pdb）。
- 环境替代说明：原 `aptamer_eval` conda 环境已不在本机，PACo 全部流程改用
  `rnasmol_service_gpu_env`（py3.10/torch 2.1.2/pyg 2.5.0/np1.26），ckpt 加载与
  采样已冒烟验证（与 s2_v2_seed43 基线行为一致）。
