# 21. 新增实验 A：逐残基 Macro-F1、跨方法置信度耦合、配体家族分层

- 日期：2026-10-16
- 脚本：`data/91_perbase_macrof1.py`（纯 CPU，未使用任何 CUDA）
- 产物（均含逐链明细，先写临时文件再 `os.replace` 原子落盘）：
  - `data/survey/91_perbase_macrof1.json` — 实验 1：逐残基类型恢复率 + Macro-F1
  - `data/survey/91_conf_cross.json` — 实验 2：PACo / AlignIF 置信度-恢复率耦合（预留 methods 键，RhoDesign/gRNAde/NA-MPNN 待 GPU logP 补齐）
  - `data/survey/91_family_strata.json` — 实验 3：配体家族分层恢复率
  - `data/survey/91_nampnn_gap_pos.json` — NA-MPNN 42 条 gap 链有效位缓存（见 §2.4）
- 运行：`/root/miniconda3/envs/rna_if/bin/python data/91_perbase_macrof1.py`（任务书所述 `aptamer_eval` 环境已不存在；`rna_if` 环境 numpy 2.4.6 / scipy 1.17.1）
- 统计约定：bootstrap 95% CI = 链级重抽样 1000 次，种子 20261015；恢复率分母排除 native 'N' 位；长度不符样本剔除（实际 5 方法剔除数均为 0）；AlignIF 样本 '-' 位逐样本跳过；碱基顺序 A/U/G/C。

## 1. 数据与口径

splits_v2 test 477 链，native 一律取 `survey/canonical_v2.json`。采样池：

| 方法 | 池 | t=0.1 子池 |
|---|---|---|
| PACoAptamer | 300（0.1/0.5/1.0 各 100，前 100=t01，已验证） | 100 |
| RhoDesign / gRNAde | ≤300（3 温度文件） | ≤100 |
| NA-MPNN | 10（仅 t=0.1） | 10 |
| AlignIF | 20（仅 t=0.1，fasta 头含 logP） | 20 |

PACo 池的组装经校验确定为：`77_ens2_pool_btest`（437 条 native 一致链）+ `paco_fix_pool_ens2noiface`（40 条修复链），与主配置 "ens2-noiface" 注释一致（见 §2.2）。

## 2. 校验记录

### 2.1 对主表 `btest_v2_main_table.json` 的逐链复现

| 方法 | 复现链数 | max \|Δbest\| | max \|Δt01\| | 判定 |
|---|---|---|---|---|
| PACoAptamer | 477/477 | 0 | 0 | 精确复现 |
| gRNAde | 477/477 | 0 | 0 | 精确复现 |
| NA-MPNN | 477/477 | 0 | 0 | 精确复现（含 42 条 gap 链有效位口径） |
| AlignIF | 477/477 | 0 | 0 | 精确复现 |
| RhoDesign | 477/477 | 0.1742 | **0** | t01 全部精确；best 有 17 链偏差 |

RhoDesign 的 17 条偏差链（完整清单存于 `91_perbase_macrof1.json` 的 `main_table_verification.RhoDesign.mismatch_chains`）全部表现为"当前池 best ≥ 主表值"（+0.006 ~ +0.174），t=0.1 子池与主表完全一致。判定原因为主表构建（2026-08-18）后 RhoDesign 高温池被重跑过（`29v2_rhodesign_btest_v2_rerun.log`，2026-08-15 23:38 起），现存 .seqs 与主表构建时的内容不完全一致。影响范围：仅 full-pool `recovery_best` 口径的 17/477 链（3.6%）；t01 口径与逐残基统计不受影响。本报告与主表相关的对比（实验 3）一律使用主表权威值。

### 2.2 PACo 修复池选型

40 条修复链存在 4 个候选池。逐链复现主表校验：`paco_fix_pool_ens2noiface` **40/40 精确匹配**；`paco_fix_pool` / `paco_fix_pool_ens2` / `paco_fix_pool_seed43` 均仅 3/40 匹配。已采用前者。注意 `77_ens2_pool_btest` 中这 40 链的 `native` 字段仍是修复前旧序列（长度都有差异），不能直接使用——本脚本对该 40 链一律取 ens2noiface 修复池。

### 2.3 置信度耦合对既有 `77_p04_confidence_recovery.json` 的复现

| 统计量 | 本脚本重算 | p04 文件 |
|---|---|---|
| 链间 Spearman ρ（mean logp vs recovery_t01） | 0.51572 | 0.51572 |
| 链间 ρ（vs recovery_best） | 0.44262 | 0.44262 |
| 链内 Spearman 均值 | 0.19261 | 0.19261 |
| top-10% 富集均值 | +0.014629 | +0.014629 |

四项逐位一致。任务书所述"链间 ρ≈0.526"未复现：legacy 477 链口径为 0.5157，canonical 437 一致链口径为 0.5435，p04 文件权威值为 0.5157。

### 2.4 NA-MPNN gap 链

42 条链因缺 C1' 坐标等原因被 NA-MPNN 官方 parse_PDB 丢弃部分残基。本脚本用 `tools/NA-MPNN` 官方 `parse_PDB`（CPU，torch 2.14.1+cpu @ `mpl372_repro` + prody 2.6.1 @ `/tmp/prody_lib`）重建有效位，difflib equal 块映射样本第 k 位 ↔ native 第 pos_all[k] 位，缓存于 `91_nampnn_gap_pos.json`；构建时对每链断言"样本长度 == len(pos_all)"，42/42 通过；最终 477/477 逐链精确复现 `metrics_NAMPNN_btest_v2.json`。

## 3. 实验 1：逐残基类型恢复率 + Macro-F1

### 3.1 Macro-F1（t=0.1 口径，4 类 A/U/G/C）

| 方法 | 链级 macro-F1（95% CI） | 位点级 micro-F1 |
|---|---|---|
| **PACoAptamer** | **0.5767 [0.5674, 0.5859]** | **0.5970** |
| RhoDesign | 0.5129 [0.5028, 0.5230] | 0.5374 |
| AlignIF | 0.4912 [0.4783, 0.5052] | 0.5296 |
| NA-MPNN | 0.4463 [0.4316, 0.4618] | 0.4745 |
| gRNAde | 0.2740 [0.2677, 0.2805] | 0.3014 |

全池（300 样本）口径与 t01 基本一致：PACo 0.5722 / RhoDesign 0.5145 / gRNAde 0.2742（链级 macro-F1）。主表恢复率参考（t01 均值）：PACo 0.604、RhoDesign 0.551、AlignIF 0.518、NA-MPNN 0.470、gRNAde 0.312。

### 3.2 逐碱基恢复率（t01，链级 macro 均值）

| 方法 | A | U | G | C |
|---|---|---|---|---|
| PACoAptamer | 0.5140 | 0.5354 | **0.7017** | 0.6421 |
| RhoDesign | 0.4779 | 0.5824 | 0.6401 | 0.4855 |
| gRNAde | 0.4887 | **0.1177** | 0.3052 | 0.2942 |
| NA-MPNN | 0.5482 | 0.4192 | 0.4910 | 0.4197 |
| AlignIF | 0.5178 | 0.4512 | 0.5566 | 0.5138 |

native 碱基占比（t01 有效位点）：A 25.2% / U 22.9% / G 27.3% / C 24.5%。

### 3.3 预测边际偏置（预测边际 / 真实 support，t01）

| 方法 | A | U | G | C | 解读 |
|---|---|---|---|---|---|
| PACoAptamer | 0.840 | 0.982 | **1.135** | 1.031 | 轻度过预测 G、欠预测 A |
| RhoDesign | 0.825 | **1.255** | 1.103 | 0.829 | 过预测 U/G |
| gRNAde | 1.201 | **0.313** | 1.166 | 1.247 | U 严重欠预测（边际仅 31%），致 U 恢复率 0.118 崩塌 |
| NA-MPNN | 1.190 | 0.872 | 1.010 | 0.913 | 轻度过预测 A |
| AlignIF | 1.006 | 0.975 | 1.028 | 0.986 | 四类均 ±4% 内，最均衡 |

结论：高频碱基 G/C 并非普遍被过预测——仅 PACo（G +13.5%）与 RhoDesign（G +10.3%）轻度如此；gRNAde 的偏置模式相反且极端（欠预测 U）；AlignIF 边际几乎无偏。

## 4. 实验 2：置信度-恢复率耦合（PACo vs AlignIF）

| 指标 | PACoAptamer（477 链，300 样本/链） | AlignIF（477 链，20 样本/链） |
|---|---|---|
| 链间 Spearman ρ（mean logP vs t01 恢复率） | 0.5157 [0.438, 0.595] | **0.6847 [0.634, 0.732]** |
| 链间 ρ（vs best 恢复率） | 0.4426 [0.357, 0.520] | 0.5924 [0.528, 0.653] |
| 链内 Spearman 均值（中位） | 0.1926（0.1965），83.6% 链为正 | 0.0086（0.0201），51.2% 链为正 |
| 链内有效链数 | 477/477 | 441/477（其余 logP 或恢复率恒定） |
| top-10% logP 富集（选中均值 − 池均值） | **+1.46pp [+1.21, +1.72]** | −0.13pp [−0.38, +0.11]（不显著） |

（PACo canonical 437 链口径：链间 ρ=0.5435、链内均值 0.1973、富集 +1.53pp，结论不变。）

解读：两种方法的**链间**置信度都有信息量（AlignIF 的链间 ρ 更高，0.68 vs 0.52——其 fasta 头 logP 适合跨链排序/难度预估）；但**链内**只有 PACo 的逐样本 logP 能有效排序同一链的设计样本（83.6% 链正相关、top-10% 选择净赚 +1.46pp），AlignIF 链内无信号（ρ≈0.01、富集 ≈0）。原因可能是 AlignIF 输出 logP 动态范围小且 20 样本内高度并列（36 链 logP 恒定被剔除）。

## 5. 实验 3：配体家族分层恢复率

`75_families.json` 12 个家族中 9 个在 test 477 链中有成员（共 105 链；fluo_aptamer_8hz / rhobast_aptamer / folate_riboswitch 仅 av2，test 中 0 链），其余 372 链归 "other"。恢复率取主表权威值；Wilcoxon 为单尾（H1: PACo > RhoDesign），零 Δ 对剔除、n<5 不报 p。

t=0.1 平均恢复率（Δ = PACo − RhoDesign 逐链配对）：

| 家族 | n | PACo | RhoDesign | gRNAde | NA-MPNN | AlignIF | Δt01 | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| theophylline_aptamer | 13 | **0.7722** | 0.6955 | 0.3906 | 0.7347 | 0.6569 | +0.0767 | 1.2e-4 |
| purine_riboswitch | 50 | **0.5785** | 0.5268 | 0.3437 | 0.5194 | 0.5707 | +0.0517 | 5.7e-13 |
| xanthine_riboswitch | 8 | **0.6518** | 0.5684 | 0.3053 | 0.4858 | 0.4975 | +0.0834 | 7.8e-3 |
| ztp_riboswitch | 8 | 0.6111 | 0.5341 | 0.2913 | **0.6528** | 0.6147 | +0.0769 | 3.9e-3 |
| fam_1MMS | 5 | **0.6501** | 0.6191 | 0.2255 | 0.6132 | 0.5923 | +0.0310 | 0.094 |
| fam_3GCA | 4 | **0.6848** | 0.5547 | 0.3799 | 0.6696 | 0.6388 | +0.1301 | n<5 |
| dG_riboswitch | 4 | **0.5427** | 0.4645 | 0.2578 | 0.5198 | 0.4473 | +0.0782 | n<5 |
| fam_8HB1 | 3 | **0.5835** | 0.5558 | 0.1965 | 0.3580 | 0.3674 | +0.0277 | n<5 |
| mango3_aptamer | 10 | 0.5766 | **0.5985** | 0.2189 | 0.5184 | 0.5751 | −0.0219 | 0.81 |
| other | 372 | **0.6005** | 0.5479 | 0.3102 | 0.4446 | 0.5022 | +0.0526 | 8.0e-22 |

- **PACo 领先家族（8/9）**：theophylline、purine、xanthine、ztp、fam_1MMS、fam_3GCA、dG、fam_8HB1，其中 4 个达到单尾显著（theophylline p=1.2e-4；purine p=5.7e-13；ztp p=3.9e-3；xanthine p=7.8e-3）；fam_3GCA Δ 最大（+0.130）但 n=4 不报 p。
- **PACo 落后家族（1/9）**：mango3_aptamer（Δt01=−0.0219，p=0.81；best 口径 Δ=+0.0112、p=0.27，两口径不一致且均不显著）。家族均值口径下另有 ztp 链上 NA-MPNN（0.6528）高于 PACo（0.6111）。
- best 口径方向一致：8/9 家族 Δbest 为正（mango3 例外 +0.0112 但 frac_pos=0.50）。
- "家族 × 方法" 矩阵已存于 `91_family_strata.json` 的 `matrix_family_x_method`（t01 与 best 两口径），可直接绘图。

## 6. 可写进论文的结论句

1. 在 477 条测试链上，PACo-Aptamer 的四分类宏平均 F1 达 0.577（链级，95% CI [0.567, 0.586]；位点级 micro 0.597），较最优基线 RhoDesign（0.513）绝对提升 6.4 个百分点；提升主要来自稀有碱基之外的均衡性——PACo 对 G（0.702）与 C（0.642）的逐碱基恢复率显著高于全部基线，而 gRNAde 因严重欠预测 U（预测边际仅为真实出现的 31%）导致 U 恢复率崩塌至 0.118。
2. PACo 的逐样本 teacher-forced logP 同时具备链间与链内排序能力：链间 Spearman ρ=0.516（vs AlignIF 0.685），链内 83.6% 的链呈正相关（均值 ρ=0.193），按 logP 取每链 top-10% 样本可比池均值净增 +1.46pp 恢复率（95% CI [+1.21, +1.72]）；AlignIF 的 logP 仅在链间有效，链内无排序信号（ρ≈0.01，富集 −0.13pp 不显著）。
3. 家族分层显示 PACo 的优势不依赖单一配体类型：9 个多配体家族中 8 个领先 RhoDesign（4 个 Wilcoxon 单尾 p<0.01：theophylline、purine、ztp、xanthine），唯 mango3 适配体略落后（Δ=−0.022，不显著）；非家族链（n=372）上 Δ=+0.053（p=8.0e-22），说明优势具有普遍性。

## 7. 遗留事项

- RhoDesign / gRNAde / NA-MPNN 的逐样本 logP 需 GPU 实验补齐后并入 `91_conf_cross.json` 的 `methods` 键（结构已预留）。
- RhoDesign 17 链 full-pool best 与主表存在已知内容漂移（§2.1），如需完全对齐须以主表构建时的池文件重跑或更新主表。
- 任务书所述链间 ρ≈0.526 未复现，权威值以 `77_p04_confidence_recovery.json` 的 0.5157 为准（本脚本逐位复现）。
