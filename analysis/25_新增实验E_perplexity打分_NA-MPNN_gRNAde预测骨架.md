# 新增实验 E：native perplexity 打分 + NA-MPNN/gRNAde 预测骨架推理

> 日期：2026-10-16。执行环境：GPU 2（全程 `CUDA_VISIBLE_DEVICES=2`）。
> 三项任务：(A) NA-MPNN 在 RhoFold+ 预测骨架上批量推理 + 效率探针；
> (B) 各方法 native 序列 teacher-forced 打分（perplexity 主表列 + t=0.1 池逐样本 logP）；
> (C) gRNAde 在预测骨架上批量推理 + 效率探针。

## 〇、环境变更记录（重要）

任务书指定解释器 `/root/miniconda3/envs/aptamer_eval/bin/python` **在当前机器已不存在**
（`/root/miniconda3/envs/` 下无 aptamer_eval；历史 btest 日志为 2026-08 月用该环境跑出）。
替代方案（不动任何共享服务环境）：

| 用途 | 解释器 | 补充依赖 |
|---|---|---|
| gRNAde / NA-MPNN | `/root/miniconda3/envs/odesign_service_env/bin/python`（torch 2.3.1+cu121, pyg 2.7.0, torch_cluster/scatter 可用, biotite/yaml/pandas/Bio/wandb 齐备） | `data/survey/pylibs/`（pip --target 安装 prody 2.6.1，NA-MPNN 用）；`data/survey/pyshims/cpdb.py`（gRNAde data_utils 顶层 import 的占位 stub，featurize_custom 路径不调用 cpdb.parse） |
| RhoDesign | `/root/miniconda3/envs/rhodesign_service_env/bin/python`（torch 1.12.1, pyg 2.5.0，29v2 btest 同款） | 无 |
| RNAfold | `/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold --noPS` | — |

gRNAde 冒烟（模型加载+featurize+t=0.1 采样）与 NA-MPNN 冒烟（design 模式跑通 .fa 输出）均验证通过后才进入批量。

## 一、口径校验记录

恢复率口径与 `data/survey/btest_v2_main_table.json` per_chain 的一致性核对：
随机抽 10 条 test 链（seed=0），分别用晶体骨架既有输出重算
（gRNAde：`grnade_out_btest_v2` 三温度池 best + t=0.1 池 mean；NA-MPNN：`nampnn_out_btest_v2`
10 样本池同口径），与主表 per_chain 逐链比较。

结果：**10/10 链 × 2 方法，recovery_best 与 recovery_t01_mean 全部逐位一致（差 < 1e-9）**。
口径确认：native = `canonical_v2.json`；分母排除 'N' 位；best = 池内最大同一性；
t01 mean = t=0.1 池逐样本平均；长度不符样本过滤。后续 102/103 号指标均按此口径。

## 二、重要发现：gRNAde btest 输出的 C/G 互换问题

任务 C 前置核查中发现：**26v2 推理脚本（data/26v2_run_grnade_btest.py）解码采样 token 用的
`vocab=["A","C","G","U"]` 与模型真实 token 序不符**。

证据链：
1. 训练管线：`47_launch_grnade_btrain_v2.sh` → 官方 `tools/gRNAde/main.py` →
   `src/data/dataset.py` → `RNAGraphFeaturizer`，其 `letter_to_num` 来自
   `src/constants.py: RNA_NUCLEOTIDES = ["A","G","C","U"]`，即 token 序 **A=0, G=1, C=2, U=3**。
   26v2 的 vocab 把 token 1 解码成 "C"、token 2 解码成 "G"——C/G 互换。
2. Teacher-forced native logP 裁决（btrain_v2 ckpt，5 链）：
   官方序打分全部 5 链显著更高（如 1D4R_A：−0.970 vs −1.121 nat/nt）。
3. 40 链 × 100 样本（t=0.1）配对对比：官方解码逐链恢复率均值高 +0.0199
   （官方胜 25 链 / 26v2 胜 12 链 / 平 3）。
4. 用既有 .seqs 字符串做确定性 C↔G 互换即得官方解码序列（映射是双射），
   重算 477 链晶体骨架恢复率：

| 口径 | recovery_best | recovery_t01_mean |
|---|---|---|
| 主表现值（26v2 解码，C/G 互换） | 0.5015 | 0.3121 |
| 官方解码修正后 | **0.5300** | **0.3577** |
| 差值 | +2.85pp | **+4.56pp** |

193 链（86_sctm）子集同样成立：26v2 解码 0.4946/0.3204 → 官方解码 0.5223/0.3637。
重算证据存 `data/survey/98_grnade_decode_check.json`。
注意：本报告**未改动主表**；任务 C 的预测骨架输出已按官方序解码，
与晶体口径的 Δ 对比以"官方解码修正后"为基准（另附 26v2 口径对照）。

## 三、任务 B：native 序列 teacher-forced 打分（perplexity）

脚本：`data/98_native_score.py`；产物：`data/survey/98_part_{grnade,nampnn,rhodesign}.json`
→ 汇总 `data/survey/98_perplexity.json`。
口径：mean logP/nt（自然对数）→ `ppl_native = exp(−mean logP/nt)`；
native 非 ACGU 位（canonical 'N' 等）用占位 token 喂入、剔除出平均；
`t=0.1` 池逐样本 logP 用同一有效位掩码（供跨方法置信度分析）。

各方法实现：
- **gRNAde**（477 链，晶体骨架三件套 `grnade_inputs_btest_v2`，btrain_v2 ckpt）：
  `model.forward(batch)` 直接返回逐位 logits（官方 teacher-forced 路径）；
  池打分用 PyG Batch 每 50 样本一批。
- **NA-MPNN**（477 链，`canonical_v2` PDB，s_11016.pt）：run.py 无 score 模式 →
  复用官方 `model.sample` 路径做 teacher-forced：`chain_mask=1`（使 log_probs 被记录）
  + bias 钉死（真值位 0、其余位 −1e9，multinomial 必取真值 token），
  条件上下文即真值前缀；断言输出 S==目标序列防漂移。链联合似然与解码序无关（链式法则）。
  缺 C1' 坐标被 parse 丢弃的链按 61 号"有效位口径"（difflib 等块对齐，仅在可信 ACGU 位上平均）。
- **RhoDesign**（477 链，btest 同款 PDB 解析顺序 fix4→chain_pdb→chain_pdb_v2，best_val.pth）：
  27v2 训练同款 teacher-forced forward（decoder 输入 tokens[:-1]，目标 tokens[1:]），
  忽略 padding 与 X 目标位。
- **PACo**（对照，免重跑）：`77_ens2_pool_btest/*.json` 的 `logp_native`
  同为 teacher-forced 逐位平均自然对数（集成概率平均口径），ppl = exp(−logp_native)。

### Perplexity 汇总（宏平均，477 链）

| 方法 | n | logp_native 均值 | **ppl_native 宏平均** | t=0.1 池样本 logP 均值 |
|---|---|---|---|---|
| PACoAptamer（ens2，对照） | 477 | −0.9975 | **2.911** | （77 号另有逐样本 logp） |
| gRNAde | 477 | −1.0970 | **3.016** | −0.8959 |
| NA-MPNN | 477 | −1.1589 | **3.401** | −0.3407 |
| RhoDesign | 477 | −1.1494 | **3.450** | −0.2442 |

读法提示：ppl 越低说明模型越"认可" native 序列。三方法对 t=0.1 自采样本的打分均高于
native（mode-seeking 采样集中于高置信区）；RhoDesign t=0.1 池几乎无多样性
（如 1D4R_A 池 100/100 条完全相同），故其样本 logP 均值（−0.244）显著高于他者；
PACo 的 native 似然最高，与其最高恢复率一致。

## 四、任务 A/C：预测骨架（RhoFold+，86_sctm 193 链）推理汇总

输入：193 条 test 链的 RhoFold+ 预测结构 `data/survey/86_sctm/native_{stem}.pdb`
（链 ID 0、无 HETATM、残基数与 native 序列等长——193/193 全部通过核查）。

| 方法 | 骨架 | n | recovery_best | recovery_t01_mean | Δ best vs 晶体 | Δ t01 vs 晶体 |
|---|---|---|---|---|---|---|
| NA-MPNN | 晶体（61 号口径，193 子集） | 193 | 0.5436 | 0.4947 | — | — |
| NA-MPNN | **RhoFold+ 预测（102）** | 193 | **0.3528** | **0.2964** | **−0.1908** | **−0.1983** |
| gRNAde | 晶体（官方解码修正，193 子集） | 193 | 0.5223 | 0.3637 | — | — |
| gRNAde | **RhoFold+ 预测（103）** | 193 | **0.5066** | **0.3500** | **−0.0157** | **−0.0137** |
| （参考）gRNAde | 晶体（主表现值，26v2 解码） | 193 | 0.4946 | 0.3204 | | |

要点：
- NA-MPNN 对预测骨架非常敏感（Δ≈−19pp）：纯结构条件化，骨架误差直接传导。
- gRNAde 几乎不受骨架误差影响（Δ≈−1.5pp）：其 3D dropout 训练 + 外部 SS 输入
  （RNAfold 对 native 序列的预测，与晶体口径同协议）提供了稳定条件信号。
- 两方法 193/193 全覆盖，无失败链；NA-MPNN 无 gap 链（预测骨架原子齐全）。
- 口径说明：103 号输出按官方 token 序解码（修正 26v2 的 C/G 互换，见第二节）；
  Δ 以官方解码修正后的晶体值为基准。若与主表现值直接比，gRNAde 预测骨架
  反而"高"于晶体（0.5066 vs 0.4946），这是解码 bug 的假象，不是真实物理结论。

### 效率探针（GPU 2，前 6 链）

| 方法 | 链长 | 墙钟 s | 显存峰值 MiB | 备注 |
|---|---|---|---|---|
| NA-MPNN | 26–127 | 3.4–3.8 | 560–802 | t=0.1×10 单次 batch；显存为 nvidia-smi 0.2s 轮询差值（近似） |
| gRNAde | 26–127 | 2.7–9.6 | 69–631 | featurize+三温度采样全程；torch.cuda.max_memory_allocated 精确值 |

详表：`data/survey/99_efficiency_NA-MPNN.json`、`data/survey/99_efficiency_gRNAde.json`。

## 五、产物清单

| 路径 | 内容 |
|---|---|
| `data/102_run_nampnn_predbb.py` | 任务 A 脚本（输入链接→推理→探针→指标一体） |
| `data/103_run_grnade_predbb.py` | 任务 C 脚本（三件套准备→推理→探针→指标一体） |
| `data/98_native_score.py` | 任务 B 脚本（--method grnade/nampnn/rhodesign/--merge） |
| `data/survey/102_predbb_inputs/` | NA-MPNN 预测骨架输入符号链接（193） |
| `data/survey/nampnn_out_predbb/` | NA-MPNN 预测骨架输出池（193×.seqs，各 10 样本 t=0.1） |
| `data/survey/102_nampnn_predbb.json` | NA-MPNN 预测骨架逐链+汇总指标 |
| `data/survey/grnade_inputs_predbb/` | gRNAde 预测骨架三件套（193×3） |
| `data/survey/grnade_out_predbb/` | gRNAde 预测骨架输出池（193×{0.1,0.5,1.0}.seqs + .native） |
| `data/survey/103_grnade_predbb.json` | gRNAde 预测骨架逐链+汇总指标 |
| `data/survey/98_part_{grnade,nampnn,rhodesign}.json` | 三方法逐链打分（含逐样本 logP） |
| `data/survey/98_perplexity.json` | perplexity 汇总（{method: {stem: {...}}} + macro_summary） |
| `data/survey/98_grnade_decode_check.json` | C/G 互换修正的晶体口径重算证据 |
| `data/survey/99_efficiency_{NA-MPNN,gRNAde}.json` | 效率探针（各 6 链） |
| `data/survey/pylibs/`、`data/survey/pyshims/` | 环境替代依赖（prody pip-target / cpdb stub） |

## 六、失败与遗留

- 任务 B 三方法全部 477/477 成功（NA-MPNN 中途修复一处：修饰残基位 parse 为 UNK/DX/RX 时
  sample() 将对应概率行置零，bias 钉死会导致 multinomial 全零行报错——无效位改钉 'A' 占位，
  该位本就在有效位掩码外；修复后 5NFV_B 等链通过）。
- 主表 gRNAde 行仍为 26v2 解码（C/G 互换）口径，如需修正主表，用
  `98_grnade_decode_check.json` 的重算值或对 `grnade_out_btest_v2/*.seqs` 做 C↔G 互换后重跑 63 号。
  是否改主表由项目决定，本批实验未动既有文件。
