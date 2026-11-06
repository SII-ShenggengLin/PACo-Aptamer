# 26. gRNAde 解码 C/G 互换修正与全指标重算报告

> 日期：2026-10-16。执行：本修正任务（104 系列脚本）。
> 前置：bug 发现与独立验证见 `analysis/25_新增实验E_perplexity打分_NA-MPNN_gRNAde预测骨架.md` §二
> 与 `data/survey/98_grnade_decode_check.json`。

## 一、bug 证据复述

**现象**：gRNAde 的 B 层（btest_v2，477 链）与 A 层（av2_retrained，289 链）采样输出
（`data/survey/grnade_out_btest_v2/`、`grnade_out_av2_retrained/` 的 `.seqs`）写盘时
把碱基 C 与 G 解反了。

**机制**：推理脚本 `data/26v2_run_grnade_btest.py:116`（btest）与 `data/17v2_run_grnade.py:107`
（av2，同一脚本族）用 `vocab = ["A","C","G","U"]` 把采样 token 解码成字符；而 gRNAde
训练管线的真实 token 序是 `tools/gRNAde/src/constants.py: RNA_NUCLEOTIDES = ["A","G","C","U"]`
（A=0, G=1, C=2, U=3）。token 1（实为 G）被写成 "C"，token 2（实为 C）被写成 "G"，
A/U 不受影响。

**独立验证（25 号报告 §二，btrain_v2 ckpt）**：
1. Teacher-forced native logP 裁决：官方序（A,G,C,U）解码在 5/5 抽查链上显著更优
   （如 1D4R_A：−0.970 vs −1.121 nat/nt）。
2. 40 链 × 100 样本配对对比：官方解码逐链恢复率均值高 +0.0199（25 胜 / 12 负 / 3 平）。
3. 映射是确定性双射 → 对既有 `.seqs` 字符串做 `str.maketrans('CG','GC')` 即得官方解码
   序列；重算 477 链：best 0.5015→**0.5300**、t01 0.3121→**0.3577**
   （`98_grnade_decode_check.json`，与本次 104a 重算逐位一致，差 < 1e-12）。

**影响范围**：凡直接读取上述两个目录 `.seqs` 的指标全部受影响；native 文件为真值序列
（不经解码）不受影响；RhoFold+/USalign/barnaba 结构指标（86_sctm/92_scmcc）因输入序列
错误而受影响，需重预测。PACo/RhoDesign/NA-MPNN/AlignIF 的输出与任何真值数据不受影响。

## 二、修正方法

- 修正池（临时目录 + rename 建成，既有目录未动）：
  - `data/survey/grnade_out_btest_v2_cfix/`：477 链 × 3 温度 `.seqs`（C↔G 互换）
    + 477 个 `.native`（原样复制）。校验：修正池按 63 号口径重算 best/t01 与
    `98_grnade_decode_check.json` 的 official 列逐位一致。
  - `data/survey/grnade_out_av2_retrained_cfix/`：289 链（indep 235 ∪ zs 93）同法。
- 全部指标重算脚本为 `data/104{a..o}_*.py`，输出一律 `_gfix.json`（临时文件 + rename），
  **不覆盖任何既有文件**；每个 `_gfix.json` 只替换 gRNAde 行/块，其余行从原 JSON
  原样拷贝，并附 `cfix_note` 溯源。
- 各脚本严格复刻原脚本口径，并带自检：104i（77_p02）用同管线重算 PACo_ens2 行与原
  JSON 逐键一致（PASS）；104k（91）对 63_gfix 主表逐链复现校验 max|Δ|=0（PASS）；
  104h（82）断言 GC 含量逐链不变（C↔G 互换保持 G+C 计数，max|Δ|=0）。

## 三、逐指标 old → new 对照

### 1. 主表 gRNAde 行（63 号，477 链；→ `data/survey/63_main_table_gfix.json`）

| 口径 | old（bug） | new（cfix） | Δ |
|---|---|---|---|
| recovery_best mean | 0.5015 | **0.5300** | +2.85pp |
| recovery_best median | 0.4930 | **0.5238** | +3.08pp |
| recovery_t01 mean | 0.3121 | **0.3577** | +4.56pp |

修正后 gRNAde 仍是 477 链主表第 5 名（PACo 0.7490/0.6040 > RhoDesign 0.6936/0.5510 >
AlignIF 0.5620/0.5177 > NA-MPNN 0.5221/0.4698 > gRNAde 0.5300/0.3577）——best 口径下
gRNAde 与 NA-MPNN 的差距从 2.06pp 缩到 **−0.79pp（gRNAde 反超 NA-MPNN 的 best mean）**；
t01 口径仍末位。

### 2. 配对 Wilcoxon PACo vs gRNAde（81 号；→ `data/survey/81_wilcoxon_gfix.json`）

| 视图 | 口径 | Δ old→new | 胜率 old→new | 单侧 P old→new |
|---|---|---|---|---|
| B_test_477 | best | +0.2475 → **+0.2190** | 98.95% → **99.37%** | 9.1e-80 → 7.3e-80 |
| B_test_477 | t01 | +0.2919 → **+0.2463** | 98.95% → **99.37%** | 6.5e-80 → 6.0e-80 |
| A_indep_235 | best | +0.2527 → **+0.2316** | 98.30% → **98.72%** | 4.2e-40 → 2.3e-40 |
| A_indep_235 | t01 | +0.2804 → **+0.2587** | 99.57% → **98.72%** | 1.3e-40 → 2.0e-40 |
| A_zs_93 | best | +0.2542 → **+0.2029** | 100% → **100%** | 2.8e-17（不变量级） |
| A_zs_93 | t01 | +0.3165 → **+0.2365** | 100% → **98.92%** | 2.8e-17 → 2.9e-17 |

PACo 对 gRNAde 的全部领先主张（三组 × 双口径）**不变**，Δ 缩小 2–8pp 但仍极显著。

### 3. A 层主表 gRNAde 行（69 号；→ `data/survey/69_alayer_gfix.json`）

| 视图 | best old→new | t01 old→new | iface_best old→new |
|---|---|---|---|
| indep 235 | 0.4917 → **0.5129** | 0.3021 → **0.3238** | 0.8765 → **0.7974** |
| zs 93 | 0.5065 → **0.5578** | 0.3116 → **0.3917** | 0.8006 → **0.8079** |
| union 289 | 0.4956 → **0.5237** | 0.3029 → **0.3389** | 0.8248 → **0.8111** |

注：indep 的 iface_best 下降（0.8765→0.7974）是修正的真实效应——旧解码下"界面恢复率
虚高"是 C/G 互换碰巧命中所致；A 层 gRNAde 排名不变（仍末位）。

### 4. 77_p02 距离梯度（115 界面链；→ `data/survey/77_p02_gfix.json`）

| 分箱 | mean old→new | norm(bin−链均值) old→new |
|---|---|---|
| ≤5Å（界面） | 0.2534 → **0.2638** | −0.0494 → **−0.0767** |
| 5–8Å | 0.3204 → **0.3675** | +0.018 → +0.027 |
| 8–12Å | 0.3466 → **0.2974** | +0.042 → −0.029 |
| 12–20Å | 0.2875 → **0.3658** | −0.016 → +0.039 |
| >20Å | 0.3242 → **0.3594** | +0.021 → +0.033 |
| 界面−>20Å 配对差 | −0.0711 → **−0.0996** | — |

gRNAde 归一化界面亏口从 −4.9% 变为 **−7.7%**，仍处"逼近口袋掉精度"的基线组
（Fig4c 正文"three of the four baselines losing accuracy … −4.9 to −11.2%"的区间
变为 −7.7% 至 −11.2%，结论不变；PACo −1.8% 与 RhoDesign +0.3% 不变）。

### 5. 77_p03 SS 保真度 INF（437 链；→ `data/survey/77_p03_gfix.json`）

| 行 | INF old→new | F1 old→new | INF≥0.9 占比 old→new |
|---|---|---|---|
| gRNAde（全池 300） | 0.6267 → **0.7045** | 0.6247 → **0.7025** | 0.3248 → **0.4057** |
| gRNAde_t01 | 0.7309 → **0.7871** | 0.7295 → **0.7860** | 0.4673 → **0.5165** |

低同一性分箱（"补偿突变"叙事的直接证据）：
| ident 分箱 | INF old→new |
|---|---|
| 0.0–0.4 | 0.6193 → **0.6831**（n=100,962） |
| 0.4–0.5 | 0.7034 → **0.7673**（n=23,160） |
| 0.5–0.6 | 0.7404 → **0.8590**（n=5,211） |
| 0.6–0.7 | 0.8091 → **0.8692**（n=396） |

spearman(ident, INF)：0.1035 → 0.1265。**"gRNAde 的 SS 保真度领先"不仅成立而且
加强**：修正后 gRNAde INF 0.7045，对 NA-MPNN 0.5234 / AlignIF 0.5017 / PACo 0.2814 /
RhoDesign 0.2481 的领先从 +10.3pp 扩大到 **+18.1pp**；"低序列同一性下配对模式仍保持"
（补偿突变叙事）同样成立且加强（0–0.4 分箱 0.683）。

### 6. 82 号 GC/MFE（437 链；→ `data/survey/82_gc_mfe_gfix.json`）

| 口径 | old | new |
|---|---|---|
| GC mean | 0.587500 | **0.587500（逐位断言不变）** |
| MFE/nt mean | −0.4250 | **−0.4416** |
| MFE/nt median | −0.4557 | **−0.4765** |

GC 不变符合理论（C↔G 互换保持 G+C 计数，已逐链断言 max|Δ|=0）；MFE 变负
（修正后序列折叠更稳定）。native 参照 −0.2541、PACo −0.258 不变，故正文 Fig5e
"MFE/nt 最接近 native 统计"的排序不受影响（gRNAde 仍是最负端）。

### 7. 82 号温度-多样性（Fig5a；→ `data/survey/82_temp_diversity_gfix.json`）

| 温度 | rec old→new | div old→new | uniq old→new |
|---|---|---|---|
| t=0.1 | 0.3151 → **0.3596** | 0.1607 → 0.1607 | 0.7977 → 0.7977 |
| t=0.5 | 0.2948 → **0.3269** | 0.5204 → 0.5204 | 0.9971 → 0.9971 |
| t=1.0 | 0.2787 → **0.2998** | 0.6636 → 0.6636 | 1.0000 → 1.0000 |

多样性/唯一率不变（双射映射保持两两一致度与集合基数），只有恢复率变。
Fig5a 中 gRNAde 的 rec 曲线整体上移 ~3pp，多样性结论不受影响。

### 8. 83 号 novelty（→ `data/survey/83_novelty_gfix.json`）

| 口径 | old→new |
|---|---|
| vs train 中位 max-identity | 0.429 → **0.485** |
| vs train q90 | 0.652 → **0.667** |
| vs train >0.8 占比 | 3.28% → **1.28%** |
| vs train max | 0.968 → **0.938** |
| vs AptaDB 中位 | 0.438 → **0.457** |

修正后 gRNAde 设计与训练库的最大相似度略升（向其他方法的中位 ~0.48 靠拢），
高相似（>0.8）占比反而从 3.3% 降到 1.3%；正文 Fig5b 关于 PACo 的陈述
（中位 0.48、2.1% >0.8）不受影响，gRNAde 仍处"非记忆"区间。

### 9. 85 号家族 KL（429 家族链；→ `data/survey/85_family_kl_gfix.json`）

| 方法 | KL mean（不变行并列） |
|---|---|
| PACo | 3.7280（不变） |
| NA-MPNN | 4.3055（不变） |
| AlignIF | 4.4404（不变） |
| RhoDesign | 4.6364（不变） |
| gRNAde | 4.6097 → **4.6613**（CI95 [4.524, 4.795]） |
| random 参照 | 3.4656（不变） |

正文 Fig5d "KL 3.73 versus 4.31–4.64 for the baselines" 的区间上界变为
**4.31–4.66**；PACo 最优的排序不变。（t-SNE 坐标已重算为 `85_tsne_points_gfix.json`
——2026-10-16 由 `data/104p_85_tsne_gfix.py` 生成，60 链/6610 点计数与旧版一致，
gRNAde 池换 `grnade_out_btest_v2_cfix`，`--validate` 复现校验通过；KL 为定量口径。）

### 10. 91 号逐残基/Macro-F1（477 链；→ `data/survey/91_perbase_macrof1_gfix.json`）

| 口径 | old→new |
|---|---|
| macroF1 t01（链级 mean） | 0.2740 → **0.3132** |
| macroF1 t01（micro） | 0.3014 → **0.3374** |
| per-base t01 recall：A | 0.4887 → 0.4887（不变） |
| per-base t01 recall：U | 0.1177 → 0.1177（不变） |
| per-base t01 recall：G | 0.3052 → **0.3842** |
| per-base t01 recall：C | 0.2942 → **0.3813** |

A/U 行逐位不变（token 未受影响），G/C 行互换后吸收对方原本的召回并含配对结构变化；
主表复现校验对 63_gfix 477/477 链 max|Δ|=0。

### 11. 91 号家族分层（→ `data/survey/91_family_strata_gfix.json`）

gRNAde t01 mean old→new：theophylline 0.3906→0.5011（n=13）、purine 0.3437→0.3763
（n=50）、mango3 0.2189→0.3206（n=10）、xanthine 0.3053→0.5069（n=8）、ztp
0.2913→0.3770（n=8）、fam_1MMS 0.2255→0.3174（n=5）、fam_3GCA 0.3799→0.5050（n=4）、
dG 0.2578→0.2472（n=4）、fam_8HB1 0.1965→0.2325（n=3）、other 0.3102→0.3487（n=372）。
PACo_vs_RhoDesign 配对块不涉及 gRNAde，原样保留。

### 12. 92 号界面 logo（→ `data/survey/92_interface_logos_gfix.json`）

| 口径 | old→new |
|---|---|
| 115 链 共识=native micro | 0.2540 → **0.2725** |
| 115 链 共识=native macro | 0.2287 → **0.2417** |
| 115 链 match_frac mean | 0.2448 → **0.2421** |
| 12 案例链 micro | 0.2403 → **0.3023** |

PACo 115 链 micro 0.6000 不变，界面共识的跨方法排序不变。

### 13. scTM / scMCC（86/92 号，193 链；→ `data/survey/86_sctm_gfix.json`、`92_scmcc_gfix.json`）

193 条链的修正后 gRNAde 池内 best 设计（逐链序列与恢复率存
`data/survey/86_sctm_gfix_designs.json`）经 RhoFold+ 重预测（GPU 2，
`data/survey/86_sctm/gRNAdeCFIX_{stem}.pdb`，193/193 成功），USalign（native 在前，
-outfmt 2 第 3 列）与 barnaba（非 XXX 配对，参考 = s1 缓存 ss_pairs，论域 j−i≥4）
按 86/92 号同款口径评估：

| 指标 | old（bug 序列） | new（cfix） |
|---|---|---|
| scTM mean | 0.2193 | **0.2651** |
| scTM median | 0.2150 | **0.2348** |
| scTM ≥0.45 占比 | 0.5% | **10.4%** |
| scTM ≥0.31 占比 | 13.5% | **28.0%** |
| scMCC mean | 0.3908 | **0.4724** |
| scMCC median | 0.3760 | **0.5732** |
| PPV mean | 0.3553 | **0.4080** |
| Sens mean | 0.4173 | **0.5250** |

横向位置（修正后）：scTM native 0.4655 > NA-MPNN 0.3246 > **gRNAde 0.2651 ≈ PACo
0.2634** > RhoDesign 0.2397 > randmatch 0.2256；scMCC native 0.5458 > **gRNAde
0.4724** > NA-MPNN 0.4629 > PACo 0.3437 > RhoDesign 0.2876 > randmatch 0.2004。
（两 JSON 中另存 `gRNAde_orig_bugdecode` 行作对照；其余行原样拷贝。）

## 四、哪些论文数字会变（逐条）

- **主表（Fig. 2a,b 与正文 48 行附近表格）**：gRNAde 行 50.15%→**53.00%**（best）、
  31.21%→**35.77%**（t01）。其余行不变。注意 best 口径下 gRNAde（53.00%）将略高于
  NA-MPNN（52.21%）——若正文/表中按数值排序或点名"最弱基线"，措辞需复查。
- **Fig. 2c（配对显著性）**：PACo−gRNAde Δ 与 P 值变小但仍极显著（见 §三.2）；
  正文未点名 gRNAde 的 Δ，仅图面板数字变。
- **Fig. 2d–f（胜率散点）**：PACo vs gRNAde 胜率 98.95%→**99.37%**（best）/
  98.95%→**99.37%**（t01）；正文引用的 RhoDesign 77.1% 与 AlignIF 95.4% 不变。
- **Fig. 3（indep/zs 表与配对面板）**：gRNAde 行 best/t01/界面列全部更新（§三.3）；
  PACo 领先结论不变。
- **Fig. 4c（距离梯度）**：gRNAde 曲线与归一化亏口 −4.9%→**−7.7%**；正文
  "−4.9 to −11.2%" 区间变为 "−7.7 to −11.2%"。
- **Fig. 4d（SS 保真度）**：gRNAde INF 0.627→**0.704**（正文 59 行"places gRNAde
  first (INF 0.627)"需改为 0.704）；INF≥0.9 占比 0.325→0.406（t01 0.467→0.517）；
  低同一性分箱数字见 §三.5。
- **Fig. 4e/f（scTM 与 matched control）**：gRNAde 点 0.219→**0.265**（Fig. 4e 散点/柱状
  更新）；正文 59 行 "The matched control collapses to gRNAde's level (0.226 versus
  0.219)" **不再成立**——修正后 gRNAde（0.265）明显高于 randmatch 对照（0.226），
  该从句需改写（如 "the control falls below every retrained baseline"）；PACo−control
  = +3.8%（P = 4×10⁻⁶）本身不受影响。
- **补充（实验 B 的 scMCC，未进正文）**：gRNAde 0.391→**0.472**，设计方法中升至第一
  （原 NA-MPNN 0.463 领先）。
- **Fig. 5a（温度-多样性）**：gRNAde rec 三点上移（§三.7），多样性不变。
- **Fig. 5b（novelty）**：gRNAde 的 train/AptaDB 最大相似度分布更新（§三.8）；
  正文点名的 PACo 数字不变，"设计不是训练集记忆"的结论不变。
- **Fig. 5d（家族 KL）**：gRNAde 4.61→**4.66**；baselines 区间 "4.31–4.64"→"4.31–4.66"。
- **Fig. 5e（GC/MFE）**：gRNAde MFE/nt −0.425→**−0.442**；GC 不变。
- **界面梯度/界面 logo 补充**：§三.4/§三.12。
- **补充表（91 perbase/Macro-F1、家族分层）**：§三.10/§三.11。
- **perplexity 主表列（98 号）**：gRNAde native teacher-forced 打分不经过采样解码，
  不受影响（25 号报告口径本就正确），无需变动。

## 五、结论是否变化的判断

1. **"gRNAde 的 SS 保真度 INF 领先"还成立吗？** —— **成立，且更强**。INF
   0.627→0.704（全池）/0.731→0.787（t01），对第二名 NA-MPNN 的领先从 +10.3pp
   扩到 +18.1pp。修正后序列才是模型真实输出，其回折配对保持度更高，与该模型
   多态设计的训练目标一致。
2. **"补偿突变"叙事还成立吗？** —— **成立，且更强**。最低同一性分箱（0–0.4）的
   INF 从 0.619 升到 0.683：即使序列同一性仅 ~30%，回折配对仍高度保持，直接支撑
   "graph 特征产生保持配对模式的补偿突变"的解释。ident–INF 相关 0.104→0.126（仍弱），
   即"恢复率与 SS 保真是独立轴"的论断也不变。
3. **PACo 的全部领先主张是否都不变？** —— **是**。主表排名、三层评估集双口径
   Wilcoxon（§三.2，全部 P ≤ 2.9e-17）、胜率（≥98.7%）、家族 KL 最优、界面共识
   领先、温度-多样性权衡位置均不变；PACo−gRNAde 的 Δ 缩小 2–8pp 但量级与显著性
   不变。
4. **需要作者额外注意的两处措辞**：
   - best-of-pool 口径下 gRNAde（53.00%）略超 NA-MPNN（52.21%），若正文以
     "gRNAde 垫底"之类排序措辞需谨慎（t01 口径仍垫底）。
   - 【scTM 结论】**一处正文从句失效**：Fig. 4f 的 "the matched control collapses to
     gRNAde's level (0.226 versus 0.219)" 在修正后变为 0.226 versus **0.265**，
     对照已显著低于 gRNAde，从句须改写；但该句的核心论证（PACo 比 recovery 匹配
     对照高 3.8%、P=4×10⁻⁶）不受影响，且修正后"恢复序列携带真实折叠信息"的总
     论点反而更强（gRNAde 自身也高于对照）。scMCC 补充口径下 gRNAde 升至设计方法
     第一（0.472 > NA-MPNN 0.463），与 INF 领先互相印证——"gRNAde 领衔 SS/结构
     保真轴、PACo 领衔恢复轴"的双轴叙事整体加强；唯 scTM 上 NA-MPNN 仍第一
     （0.325 > 0.265），"NA-MPNN leads self-consistency" 不变。

## 六、产物清单

修正池：
- `data/survey/grnade_out_btest_v2_cfix/`（1908 文件 = 477×3 seqs + 477 native）
- `data/survey/grnade_out_av2_retrained_cfix/`（1156 文件 = 289×3 seqs + 289 native）

指标 JSON（均 `_gfix` 后缀，未覆盖既有文件）：
- `data/survey/63_main_table_gfix.json`（主表 gRNAde 行 + 逐链）
- `data/survey/81_wilcoxon_gfix.json`
- `data/survey/77_p02_gfix.json`、`77_p03_gfix.json`
- `data/survey/82_gc_mfe_gfix.json`、`82_temp_diversity_gfix.json`
- `data/survey/83_novelty_gfix.json`
- `data/survey/85_family_kl_gfix.json`
- `data/survey/91_perbase_macrof1_gfix.json`、`91_family_strata_gfix.json`
- `data/survey/92_interface_logos_gfix.json`
- `data/survey/69_alayer_gfix.json`
- `data/survey/86_sctm_gfix.json`、`92_scmcc_gfix.json`、`86_sctm_gfix_designs.json`
- 中间产物：`data/survey/104_fold_cache_grnade_cfix.json`（RNAfold 回折缓存，供复算）
- 结构：`data/survey/86_sctm/gRNAdeCFIX_{stem}.pdb/.fa`（193 链）

脚本：`data/104_build_cfix_pools.py`、`data/104a_63_gfix.py`、`data/104b_69_gfix.py`、
`data/104c_81_wilcoxon_gfix.py`、`data/104e_fold_gfix.py`、`data/104f_83_novelty_gfix.py`、
`data/104g_77p03_gfix.py`、`data/104h_82_gfix.py`、`data/104i_77p02_gfix.py`、
`data/104j_85_kl_gfix.py`、`data/104k_91_gfix.py`、`data/104l_92_logos_gfix.py`、
`data/104m_86_cfix_predict.py`、`data/104n_86_92_eval_gfix.py`、`data/104o_82_tempdiv_gfix.py`。
（另：为让 gRNAde `src/constants.py` 在无 wandb 环境下可导入，新增占位 stub
`data/survey/pyshims/wandb.py`；edlib 装在 `/tmp/edlib_lib`。）
