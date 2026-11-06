# 23. 新增实验（2026-10-16 批次）：PACo 预测骨架 / 配体注意力 / SS 稳健性 / AlignIF 预测骨架 / 效率探针

> 执行环境批次说明：机器重启后 **`aptamer_eval` conda 环境已丢失**（`/root` 被重置）。
> 本批次全部实验改用 **`odesign_service_env`**（python 3.10.20 + torch 2.3.1+cu121 +
> torch_geometric 2.7.0 + **真实编译版** torch_scatter 2.1.2 / torch_cluster 1.6.3 +
> biotite 1.2.0 + hydra 1.3.2；PACoAptamer 37M 参数模型加载与前向已验证）。
> AlignIF 缺的 lightning 2.2.5 / arnie 0.2.7 以 `pip install --target=data/pylibs_alignif
> --no-deps` 装在项目目录内（PYTHONPATH 前置使用），**未改动 odesign_service_env 本体**。
> RNAfold 仍为 `/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold --noPS`。
> 全程只用 GPU 0（`CUDA_VISIBLE_DEVICES=0`）。

## 0. 复现校验（换环境后的数值可信性）

### 0.1 任务 A 晶体骨架复现校验（94 号 crystal 相位）

193 链（86_sctm 链集）以 **load_s1 缓存特征（晶体骨架口径）+ ens2×300 + seed42**
在新环境完整重跑，与主表 `btest_v2_main_table.json` per_chain['PACoAptamer'] 同链对比：

| 指标 | 新环境重跑 | 主表（旧环境） | Δ |
|---|---|---|---|
| recovery best（193 链均值） | 0.7435 | 0.7426 | **+0.0008** |
| recovery t01 mean | 0.6043 | 0.6036 | **+0.0007** |
| 逐链 Spearman | best 0.915 / t01 0.993 | — | — |
| 逐链 Pearson（best） | 0.948 | — | — |

主表全 477 链参照：best 0.7490 / t01 0.6040。**校验通过**：真实 torch_scatter
编译版与旧环境 shim 的数值差异在采样噪声量级内（best 逐链 Pearson 0.95，
均值差 <0.1pp）。

### 0.2 任务 C 条件 1（standard）复现校验

见 §3 末（standard 条件 vs 77 号池逐链精确一致率 + 主表均值）。

## 1. 任务 A：PACo(ens2) 在 RhoFold+ 预测骨架上的设计（94 号）

**设计**：193 条 test 链（86_sctm 链集，全部为 splits_v2 test 子集），
ens2 主配置（s2_v2_seed43_noiface + s2_v2_ssdo15_noiface 逐步 softmax 概率平均），
3 温度（0.1/0.5/1.0）×100 = 300 样本/链，torch.manual_seed(42)。
预测骨架条件的特征 = **9 原子坐标来自 RhoFold+ 预测结构**
（`data/survey/86_sctm/native_{stem}.pdb`），native 序列（= load_s1 缓存 seq，
已抽查 30/30 与 canonical_v2、86_sctm fa 全等）与 RNAfold(native) MFE SS
（已抽查 30/30 与缓存 ss_pairs 全等）**与晶体口径完全相同** —— 唯一变量是坐标。

**产物**：`data/survey/94_predbb_pool/{stem}.json`（77 号池同格式，含逐样本 logp）、
`94_predbb_pool_crystal/`（校验池）、`data/survey/94_predbb_paco.json`（汇总）。
脚本 `data/94_predbb_paco.py`。

**结果**：

| 口径 | best | t01 mean |
|---|---|---|
| 预测骨架（193 链） | **0.6895** | **0.5181** |
| 晶体主表（同 193 链） | 0.7426 | 0.6036 |
| **Δ（预测 − 晶体）** | **−0.0531**（95%CI [−0.062, −0.043]，Wilcoxon p=1.9e-18） | **−0.0855**（95%CI [−0.097, −0.074]，p=1.1e-26） |

**恢复率 vs RhoFold+ 骨架质量（x = sctm_native，预测结构与晶体的 TM）Spearman**：

| y | ρ | p |
|---|---|---|
| 预测骨架 best | −0.213 | 0.0029 |
| 预测骨架 t01 | +0.216 | 0.0026 |
| Δbest | +0.175 | 0.015 |
| Δt01 | +0.207 | 0.0038 |

解读：PACo 在预测骨架上的性能损失显著但温和（best −5.3pp、t01 −8.6pp），
且与骨架质量仅弱相关（|ρ|≈0.2；best 甚至弱负相关——高分骨架多来自短链等
混杂，Δ 口径下为正即"骨架越好损失越小"，符合直觉但强度有限）。
**对比 AlignIF（§4）：PACo 的骨架质量鲁棒性明显更强。**

## 2. 任务 B：配体 token 注意力可解释性（96 号）

**问题**：配体 token 仅与核苷酸 token 拼接后经 encoder 联合注意、decoder
cross-attention 读取混合表示。解码界面位时模型是否（更多）在看配体 token？

**方法**：115 条界面链（74_iface_meta.json：usable 且 0<n_iface≤50），
teacher-forced 喂 native 序列，forward hook 收 decoder 3 层 cross-attention
权重（eval 模式 `need_attn=True` 类属性使模块返回权重，hook 取输出[1]，
(B, L_tgt, L_src)，头已平均），encoder 输出最后 n_lig 个位置 = 配体 token，
其注意力质量按 query 位置求和 → 逐位"配体注意力质量"，层间平均。
配体特征 = 74 号 Phase 2 同款 lig_features（元素 one-hot(5) + 到最近 C1'
RBF(16)，(n_lig,21)），推理时填真实配体。
产物：`data/survey/96_attention.json`（脚本 `data/96_attention_ligand.py`）。

**结果**（界面位均值 − 非界面位均值，跨 115 链）：

| 条件 | Δattn mean | median | 95%CI | Wilcoxon p（单侧 greater） | 正例率 |
|---|---|---|---|---|---|
| ens2 主配置 | −0.00031 | −0.00223 | [−0.0020, +0.0013] | **0.879** | 31.3% |
| shuffle 对照（配体 token 置换） | −0.00034 | −0.00220 | [−0.0021, +0.0013] | 0.888 | — |
| seed43_noiface 单模对照 | −0.00177 | −0.00390 | [−0.0043, +0.0009] | 0.99994 | — |

- **置换稳健性对照通过**：shuffle 前后总注意力质量均值绝对差 1.8e-4 ≈ 0
  （列求和指标对 token 顺序置换理论不变，实测确认）。
- **逐位注意力 vs 到配体最近原子距离 Spearman**（115 链）：mean ρ=+0.104、
  median +0.257、负相关链仅 33.0% —— 与"越近看得越多"的预期**相反**。
- 8HZD_A 案例链（A 层，L=36，54 配体原子，7 界面位）：界面位均质量 0.0403
  vs 非界面 0.0343（此链为正方向，但属个例）；逐位明细已存
  `96_attention.json → case_8HZD_A`（native/interface/dist/三条件逐位质量）。

**结论**：decoder cross-attention 层面，模型在界面位**并未**显著多看配体 token
（差值≈0 且方向略负，p≈0.88 不显著）。结合 74 号（真实配体注入对界面恢复率
提升为噪声级），配体信息即使被利用，也主要经 **encoder 混合表示**间接传递
（配体 token 与 RNA token 在 encoder 层已联合注意），而非 decoder 的显式读取。

## 3. 任务 C：SS 输入稳健性（97 号）

**设计**：test 477 链（load_s1 缓存特征，链顺序与 77 号 phase_pool 完全一致），
ens2×300 样本/链 × 4 条件（唯一变量 = ss_pairs）：
standard（缓存原 SS = RNAfold(native) MFE）/ zero（全零矩阵）/
shuffle（位置索引置换配对矩阵行列，配对总数不变、模式破坏，torch 种子 9701）/
centroid（RNAfold `-p2` 的 centroid 结构替代 MFE）。
产物：`data/survey/97_ss_robust.json` + 断点目录 `97_ss_robust_chains/`。
477 链 × 4 条件全部成功、零跳过（约 31 min，GPU 0）。

**条件 1（standard）复现校验**：best 0.7464 / t01 0.5970；
主表 0.7490/0.6040 落在 standard 的 95%CI [0.7388, 0.7543]/[0.5887, 0.6058] 内；
与 77 号池（同 seed42、同链顺序）477 链逐链比较，均值绝对差 best 0.0192
（逐链精确一致 8/477——真实 torch_scatter 与旧 shim 的浮点微差会触发
multinomial 采样分叉，逐链不完全可复现，但分布/均值层面复现通过）。

**四条件汇总（477 链）**：

| 条件 | best（95%CI） | t01 mean（95%CI） | Δbest vs standard | Δt01 vs standard |
|---|---|---|---|---|
| standard | 0.7464 [0.739, 0.754] | 0.5970 [0.589, 0.606] | — | — |
| zero（无 SS） | 0.7436 [0.736, 0.751] | 0.5925 [0.584, 0.601] | −0.0028 [−0.0055, +0.0000] | −0.0046 [−0.0062, −0.0030] |
| shuffle（模式破坏） | 0.7437 [0.736, 0.752] | 0.5911 [0.583, 0.600] | −0.0026 [−0.0057, +0.0005] | −0.0059 [−0.0077, −0.0043] |
| centroid（替代 MFE） | 0.7480 [0.741, 0.756] | 0.5970 [0.589, 0.606] | +0.0017 [−0.0009, +0.0044] | +0.00002 [−0.0008, +0.0008] |

**结论**：PACo 对 SS 输入高度不敏感——完全去除 SS（zero）或彻底破坏配对
模式（shuffle）仅降 0.3-0.6pp（best 口径 CI 触及 0）；centroid 与 MFE 无差异。
与 74 号 no_ss 消融（全局口径噪声级）一致：**模型的结构条件信息几乎全部由
3D 几何（GVP 9 原子坐标）承载，SS 配对矩阵的边际贡献可忽略**。
shuffle ≈ zero 进一步说明残存效应来自配对度统计而非具体配对模式。

## 4. 任务 D：AlignIF 在预测骨架上的批量推理（100 号）

**设计**：输入 `data/survey/100_predbb_inputs/{stem}.pdb`（符号链接 →
`86_sctm/native_{stem}.pdb`，链接名去 native_ 前缀以匹配 AlignIF 命名）；
逐链子进程 `tools/AlignIF/inference.py --n_samples 20 --temperature 0.1
--use_msta False --checkpoint models/AlignIF_btrain_v2/alignif-seed=0-epoch=79.ckpt`，
单链 timeout 1800s，workers=6，断点续跑。指标仿 62 号（canonical native 真值，
N/- 位跳过，t=0.1 池内 best/均值）。
产物：`data/survey/alignif_out_predbb/{stem}/out.fasta` +
`data/survey/100_alignif_predbb.json`。脚本 `data/100_run_alignif_predbb.py`。

**结果**：193/193 全部成功（291s，GPU 0，6 workers）。

| 口径 | best | t01 mean |
|---|---|---|
| 预测骨架（193 链） | **0.4189** | **0.3680** |
| 晶体主表（同 193 链） | 0.5763 | 0.5387 |
| **Δ** | **−0.1574** | **−0.1707** |

Spearman(sctm_native, 预测骨架 best) = **0.694**（p=4.5e-29）；
Spearman(sctm, Δbest) = 0.430（p=4.5e-10）。

**结论**：AlignIF 对预测骨架高度敏感（best −15.7pp，且强依赖骨架质量 ρ=0.69）；
PACo 损失仅为其 1/3（−5.3pp，ρ≈0.2）。低质量骨架上 AlignIF 会出现输出坍缩
（如 1D4R_A，sctm 0.33，20 样本全为 poly-U，logp −0.14 vs 晶体 −0.65）；
高质量骨架输出正常（3GOT_A，sctm 0.91，logp −0.60）。

## 5. 任务 E：效率探针（99 号）

代表链（test 477 中长度最接近目标的各一条）：6W5C_B(50) / 1Z43_A(101) /
7LMB_B(151) / 7SYR_z(188) / 2KGP_A(250) / 7R6Q_1(295)。
GPU 0 独占、单链独立计时；PACo 用晶体缓存特征，AlignIF 用 canonical_v2 晶体
PDB（子进程，输出至临时目录 `survey/99_eff_alignif_tmp/`）。
产物：`data/survey/99_efficiency_PACoAptamer.json`、`99_efficiency_AlignIF.json`。

**PACo（300 样本/链 = 3 温度×100）**：

| 链 | L | ens2 墙钟(s) | ens2 显存峰值(MB) | samples/s | s/样本 | 单模墙钟(s) | ens/单模开销比 |
|---|---|---|---|---|---|---|---|
| 6W5C_B | 50 | 0.83 | 618 | 360 | 0.0028 | 0.44 | 1.91 |
| 1Z43_A | 101 | 1.73 | 923 | 174 | 0.0058 | 0.83 | 2.07 |
| 7LMB_B | 151 | 2.47 | 1228 | 121 | 0.0082 | 1.26 | 1.95 |
| 7SYR_z | 188 | 2.99 | 1432 | 100 | 0.0100 | 1.54 | 1.94 |
| 2KGP_A | 250 | 3.99 | 1801 | 75 | 0.0133 | 2.03 | 1.97 |
| 7R6Q_1 | 295 | 4.59 | 2081 | 65 | 0.0153 | 2.39 | 1.92 |

均值：2.77 s/链、149 samples/s、峰值 ≤2.1GB；**ens2/单模开销比 ≈ 1.96**
（理想值 2.0，逐步概率平均几乎无额外开销）。

**AlignIF（20 样本/链，t=0.1，no-MStA）**：

| 链 | L | 墙钟(s) | 显存峰值近似(MB) | s/样本 |
|---|---|---|---|---|
| 6W5C_B | 50 | 11.0 | 729 | 0.55 |
| 1Z43_A | 101 | 8.4 | 832 | 0.42 |
| 7LMB_B | 151 | 9.0 | 918 | 0.45 |
| 7SYR_z | 188 | 10.3 | 905 | 0.51 |
| 2KGP_A | 250 | 7.9 | 597 | 0.40 |
| 7R6Q_1 | 295 | 10.4 | 1241 | 0.52 |

均值：9.49 s/链、2.14 samples/s、0.47 s/样本；峰值 ≈0.6-1.2GB。
注意 AlignIF 每次调用为独立子进程，墙钟含 ~7-8s 固定开销（进程启动 +
ckpt 加载 + DSSR/RibonanzaNet 侧链），链长对其影响弱；批量跑（100 号，6 worker
常驻）摊薄后实测 193 链仅 291s（≈1.5s/链）。
**每样本耗时 PACo(ens2) ≈ 9ms vs AlignIF ≈ 475ms（含启动）**——约 50 倍差，
但 AlignIF 池规模仅 20（vs PACo 300），端到端每链 9.5s vs 2.8s。

## 附：本批次新增/修改文件清单

- `data/94_predbb_paco.py`（任务 A，crystal/predbb/summary 三相位）
- `data/96_attention_ligand.py`（任务 B，main + case8hzd 两相位）
- `data/97_ss_robust.py`（任务 C；centroid 解析已修 `d=` 后缀判定 + 长度校验，
  断点续跑 `97_ss_robust_chains/`）
- `data/99_efficiency_probe.py`（任务 E）
- `data/100_run_alignif_predbb.py`（任务 D，links/run/metrics 三相位）
- `data/pylibs_alignif/`（lightning 2.2.5 + arnie 0.2.7，--no-deps 独立目录）
- 产物：survey/94_predbb_pool{,_crystal}/、94_predbb_paco.json、96_attention.json、
  97_ss_robust{,_chains}.json、100_predbb_inputs/、alignif_out_predbb/、
  100_alignif_predbb.json、99_efficiency_{PACoAptamer,AlignIF}.json
