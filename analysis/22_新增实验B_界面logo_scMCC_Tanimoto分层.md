# 22 · 新增实验 B：界面 logo / scMCC / 配体 Tanimoto 分层

日期：2026-10-16。三个纯 CPU 分析实验（无 GPU/CUDA）。

| 实验 | 脚本 | 输出 |
| --- | --- | --- |
| 4 界面位点序列 logo 频率矩阵 | `data/92_interface_logos.py` | `data/survey/92_interface_logos.json` |
| 5 scMCC（预测 3D → 二级结构 MCC） | `data/92_scmcc.py` | `data/survey/92_scmcc.json` |
| 8 zero-shot 配体 Tanimoto 分层 | `data/93_tanimoto_zs.py` | `data/survey/93_tanimoto_zs.json` |

环境：torch 侧用 `/root/miniconda3/envs/mpl372_repro/bin/python`（torch 2.14.1+cpu；
barnaba 0.1.9 经 `PIP_CONFIG_FILE=/tmp/pip_clean.conf` 装入该环境）；Tanimoto 侧用
系统 python3（rdkit 2025.09.3 + gemmi 0.6.7 + scipy 1.17.1）；RNAfold 2.7.2
（`viennarna_fold_env`，本次未触发重算）。所有输出先写 `.tmp` 再 rename。

---

## 实验 4：界面位点序列 logo 频率矩阵

### 方法
- **界面口径与 74 号完全一致**：`74_iface_meta.json` 中 `usable` 且 `0<n_iface<=50`
  的 115 条 test 链（功能配体 HETATM + 残基原子 5 Å 接触，位点索引在 native
  坐标系）。脚本内 `assert len(usable)==115` 复核通过。
- **案例链**：Fig.6a 画廊 12 条（`figures/fig6/fig6_gallery.py`）。其中
  3FU2_C（74 口径 unusable）与 6PRV_D（无功能配体条目）不在 115 链内，
  按 n_iface 最大补入 4K31_C（17 位点）、7E9I_A（13 位点）凑满 12 条，
  `provenance` 字段标注 gallery/backfill。
- **池口径**：PACo = `77_ens2_pool_btest` pool 前 100 条（t=0.1 子池，
  n_per_temp=100）；RhoDesign/gRNAde = `{stem}.0.1.seqs`（100 条）；
  NA-MPNN（~10 条）/AlignIF（~20 条）全池。长度≠native 的序列剔除。
- 每位输出 A/U/G/C 频率（对有效字母归一）、counts、native 碱基、match_frac、
  共识碱基。汇总指标："共识碱基=native" 位点占比（micro=按位点池化，
  macro=按链平均）与 match_frac 均值。
- 边界：7E9I_A 的 NA-MPNN 池 10 条序列长度 32 ≠ native 33（NA-MPNN 预处理丢
  残基），整链剔除（故案例汇总 NA-MPNN 仅 11 链/116 位点）；同链 AlignIF
  位点 11 全部 20 条为 gap '-'，计入 `n_other_letter`，该位频率记 null。

### 关键数字

案例 12 链（129 个界面位点；NA-MPNN 缺 1 链=116 位点，AlignIF 128 位点）：

| 方法 | 共识=native (micro) | 共识=native (macro) | match_frac |
| --- | --- | --- | --- |
| **PACo** | **0.713** | **0.680** | **0.661** |
| RhoDesign | 0.674 | 0.639 | 0.640 |
| AlignIF | 0.578 | 0.563 | 0.564 |
| NA-MPNN | 0.560 | 0.515 | 0.496 |
| gRNAde | 0.240 | 0.248 | 0.265 |

全部 115 界面链（905 位点；各方法覆盖 105–115 链）：

| 方法 | 共识=native (micro) | 共识=native (macro) | match_frac |
| --- | --- | --- | --- |
| **PACo** | **0.600** | **0.626** | **0.600** |
| RhoDesign | 0.596 | 0.564 | 0.564 |
| AlignIF | 0.486 | 0.470 | 0.465 |
| NA-MPNN | 0.452 | 0.428 | 0.409 |
| gRNAde | 0.254 | 0.229 | 0.245 |

### 结论句（论文可用）
- 在 115 条界面链的 905 个界面位点上，PACo 设计池的界面位点共识碱基与 native
  一致率为 60.0%（macro 62.6%），高于全部基线（RhoDesign 59.6%、AlignIF 48.6%、
  NA-MPNN 45.2%、gRNAde 25.4%）；在 Fig.6a 的 12 条案例链上优势扩大
  （71.3% vs RhoDesign 67.4%）。
- JSON 中 `case_detail[stem].per_method[method].per_pos` 为逐位 A/U/G/C 频率
  矩阵（含 native 碱基与匹配率），可直接渲染 sequence logo 图。

---

## 实验 5：scMCC（预测 3D → 二级结构配对 MCC）

### 方法
- 输入：`86_sctm/{cond}_{stem}.pdb`（RhoFold+ 预测），193 链 × 6 条件
  （native / PACo / RhoDesign / gRNAde / NA-MPNN / randmatch；NA-MPNN 缺 9 个
  PDB，逐条记录于 `summary.NA-MPNN.missing_stems`）。
- **3D→配对**：barnaba 0.1.9 `annotate`（几何近接触 + 碱基平面夹角 ≤60° +
  H-bond donor/acceptor ≤3.3 Å 判定）。主口径 = 带有效边标注（非 "XXX"）的
  配对；`all_pairs`（含 XXX 的近接触配对）作敏感性变体同步落盘。
- **参考配对集** = native 序列 RNAfold MFE 配对：193/193 链命中
  `paco_s1_v2.pt` 的 `ss_pairs` 缓存，无需现场重算。
- **MCC 论域**：`0<=i<j<L` 且 `j-i>=4`（RNAfold 最小发夹环 3）。分母为 0
  （预测或参考配对为空）记 None，逐链保留 TP/FP/FN/TN 原始计数；另给
  zero-fill（None→0）均值作稳健性。
- 统计：bootstrap CI95（1000 次，seed 0）；配对 Wilcoxon（单侧 greater +
  双侧）；每 cond 与 86 号 scTM 并排并附 Spearman(scMCC, scTM)。

### 校验
- 完整性：per_chain 193 链 × 6 cond 全覆盖；0 长度错配；PDB 提取序列与
  对应 cond `.fa` 全部一致（n_pdb_seq_mismatch_fa=0）。
- MCC 未定义 21–29 链/cond（barnaba 未检出任何有效配对，TP+FP=0），
  zero-fill 均值排序不变。

### 关键数字（主口径 = 有效标注配对）

| cond | scMCC mean [CI95] | median | zero-fill | PPV | Sens | scTM mean(86) | Spearman(·,scTM) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.546 [0.505, 0.586] | 0.659 | 0.472 | 0.504 | 0.541 | 0.466 | 0.18 |
| PACo | 0.344 [0.299, 0.389] | 0.329 | 0.301 | 0.319 | 0.355 | 0.263 | 0.55 |
| RhoDesign | 0.288 [0.244, 0.333] | 0.192 | 0.244 | 0.276 | 0.286 | 0.240 | 0.61 |
| gRNAde | 0.391 [0.342, 0.437] | 0.376 | 0.348 | 0.355 | 0.417 | 0.219 | 0.37 |
| NA-MPNN | 0.463 [0.419, 0.508] | 0.565 | 0.403 | 0.415 | 0.480 | 0.325 | 0.50 |
| randmatch | 0.200 [0.163, 0.242] | 0.047 | 0.177 | 0.193 | 0.218 | 0.226 | 0.58 |

配对比较（annotated scMCC）：

| 对比 | n | Δ mean [CI95] | Wilcoxon p（单侧/双侧） |
| --- | --- | --- | --- |
| PACo − randmatch | 160 | **+0.125 [0.092, 0.163]** | **2.3e-10 / 4.7e-10** |
| PACo − RhoDesign | 157 | +0.052 [0.005, 0.096] | 2.2e-3 / 4.5e-3 |
| PACo − native | 162 | −0.206 [−0.247, −0.165] | — / 3.8e-17 |

### 结论句（论文可用）
- 在 recovery 匹配的前提下，PACo 设计的 RhoFold+ 重折叠结构对 native 二级
  结构的保真度显著高于随机对照（scMCC +0.125，配对 Wilcoxon p=2.3e-10），
  说明 PACo 的高恢复率集中在结构关键位，而非均匀撒点。
- PACo 的 scMCC 显著高于同结构先验家族的 RhoDesign（+0.052，p=2.2e-3），
  但仍低于 native 上界（−0.206）；设计方法中 NA-MPNN（0.463）与 gRNAde
  （0.391）的 scMCC 高于 PACo——注意二者 scTM 分别 0.325/0.219，二级保真与
  三级自洽并不等价（gRNAde 配对好但整体折叠错），论文引用时应两指标并列表述。
- 敏感性：all-pairs 口径与 zero-fill 口径下 cond 排序完全一致。

---

## 实验 8：zero-shot 配体 Tanimoto 化学相似度分层

### 方法
- **链→配体**：zs 93 链 → PDB（stem 前缀）→ `candidates_ccd.json` 的
  `ligands.other`（多配体全保留，逐配体算相似度取 max）。
- **训练配体集**：`s2_samples_v2.pt` 768 链 → 510 unique PDB → other 配体并集
  = 247 种 CCD（PDB 清单缓存 `survey/93_train_pdb_ids.json`，由脚本
  `--dump-train-pdbs` 在 torch 环境生成）。
- **SMILES**：`components.cif.gz`（gemmi 流式解析 50942 blocks），优先
  `SMILES_CANONICAL`(CACTVS)，否则任一 SMILES。RDKit Morgan FP
  （radius 2, 2048 bit）→ BulkTanimotoSimilarity。
- **分层**：按 max_tanimoto 分箱 [0,0.2)/[0.2,0.4)/[0.4,0.6)/[0.6,1.0]，
  箱内各方法 recovery_best/t01 均值 + PACo−RhoDesign 逐链配对 Δ（Wilcoxon）；
  全体 Spearman（相似度, Δ）与 Spearman（相似度, 各方法 recovery）。

### 校验（映射覆盖率）
- 74 交叉校验：29 条重叠链的 chain→CCD 集合与 candidates_ccd **29/29 一致**。
- 覆盖：93 zs 链中 88 链有功能配体；5 链 PDB 无 other 配体（5NFV_B、6IV6_G、
  6VRD_B、8SWB_B、8SWC_B）；6PQ7_C 唯一配体 OXV 解析失败 → **87 链进入分层**。
- SMILES：290 个待查 CCD 全部有 SMILES 条目；RDKit 解析失败 6 个
  （6NO、6O1、B12、B1Z、HEM、OXV——金属/配位体系 SMILES），有效指纹 284
  （训练侧 242/247）。
- 训练 PDB 在 candidates_ccd 命中率 510/510，且 510/510 含 other 配体。

### 关键数字

全体 87 链：PACo 0.759 / RhoDesign 0.719 / NA-MPNN 0.578 / AlignIF 0.599 /
gRNAde 0.502（recovery_best 均值）；**Δ(PACo−RhoDesign) = +0.041
[median +0.032]，Wilcoxon p = 1.9e-8**。

分箱表（recovery_best 均值；Δ = PACo−RhoDesign 配对）：

| max_tanimoto | n | PACo | RhoDesign | gRNAde | NA-MPNN | AlignIF | Δ best [median] | p（单侧/双侧） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [0,0.2) | 4 | 0.734 | 0.707 | 0.550 | 0.578 | 0.650 | +0.027 [+0.064] | n<10 不报 |
| [0.2,0.4) | 43 | 0.790 | 0.743 | 0.516 | 0.659 | 0.646 | +0.046 [+0.042] | 2.7e-5 / 5.4e-5 |
| [0.4,0.6) | 13 | 0.770 | 0.705 | 0.515 | 0.596 | 0.574 | +0.065 [+0.065] | 7.3e-4 / 1.5e-3 |
| [0.6,1.0] | 27 | 0.710 | 0.688 | 0.465 | 0.442 | 0.528 | +0.022 [+0.016] | 1.1e-2 / 2.2e-2 |

相关分析（n=87）：
- **Spearman(max_tanimoto, Δbest) = −0.249，p = 0.020**；Δt01 更强：
  ρ = −0.401，p = 1.2e-4。
- 各方法 recovery_best 与相似度均负相关：PACo −0.459（p=7.7e-6）、
  RhoDesign −0.326、NA-MPNN −0.511、AlignIF −0.394、gRNAde −0.358
  （全体 p≤2e-3）——zs 链配体越"新"，所有方法恢复率越低（难度效应），
  但 PACo 的相对优势反而更大。
- 散点数据在 `scatter`（x=max_tanimoto, y=Δbest/Δt01, ligands），可直接作图。

### 结论句（论文可用）
- zero-shot 链配体与训练配体的最大 Tanimoto 相似度中位数仅 0.379
  （仅 1/87 链 ≥0.9），zs 集确实是化学新颖靶标。
- PACo 对 RhoDesign 的优势随配体新颖性增大而扩大：Spearman(相似度, Δ) =
  −0.249（p=0.020；t01 口径 ρ=−0.401，p=1.2e-4）；中低相似度区间
  （0.2–0.6）Δ 达 +0.046~+0.065（p≤7e-4），高相似度区间缩至 +0.022——
  与"PACo 的配体感知模块在训练分布之外仍提供增益"的叙事一致。
- 注意最低箱 [0,0.2) 仅 4 链（Δ=+0.027，不具统计功效），引用时建议合并
  叙述趋势而非逐箱断言。

---

## 文件清单
- `data/92_interface_logos.py` → `data/survey/92_interface_logos.json`
  （definition / case_detail 12 链逐位矩阵 / case_summary / all115_summary）
- `data/92_scmcc.py` → `data/survey/92_scmcc.json`
  （definition / summary 6 cond / comparisons / per_chain 193×6 含 TP/FP/FN/TN）
- `data/93_tanimoto_zs.py` → `data/survey/93_tanimoto_zs.json`
  （definition / coverage / xcheck_74 / overall / bins_table / spearman /
  scatter 87 / per_chain 93）+ `data/survey/93_train_pdb_ids.json`
  （510 训练 PDB 清单缓存）
