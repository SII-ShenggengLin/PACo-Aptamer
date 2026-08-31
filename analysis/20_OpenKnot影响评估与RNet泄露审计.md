# OpenKnot(Science 2026)论文影响评估与 RNet 筛选实验泄露审计

> 产出:2026-08-31。承接 18/19 号(六图+手稿+代码审查)之后。
> 触发:用户新增参考文献 Townley et al., *De novo design of RNA pseudoknots with deep learning*, Science 931–936(2026-08-27 刊出,Das Lab OpenKnot 挑战赛)。
> 用户指令:①Intro 加 momentum 句;②Discussion 加 SS-only 边界段;③补参考文献(OpenKnot/RibonanzaNet/Struct2SeQ/Favor RFDdiff);④评估把 RNet 化学映射信号加入 Fig6e 筛选面板——**前提是排查信息泄露,有泄露就不加**。
> 最终状态:①②③ 完成;④ **审计发现实质泄露(FAIL),实验否决未运行**;泄露审计本身作为方法论发现写入 Discussion(一句,可删)。

---

## 一、OpenKnot 论文要点(与本项目的关系)

- **内容**:57 个假结靶标四轮盲测,AI(NA-MPNN 定骨架/RFdiff 版、gRNAde(+no3d 变体)、Struct2SeQ 深度 Q 强化学习、Rosetta、3DRNA)对比 Eterna 资深人类设计师;SHAPE 化学映射(~5 万条)+ 补偿突变 M2R-seq + cryo-EM 三层湿验证。AI 一年内追平人类,>95% 靶标可解;关键使能 = **RNet**(RibonanzaNet,~1100 万参数,训练于 ~200 万条 Eterna/RMDB 化学映射测量)做设计前瞻过滤。最佳设计 cryo-EM 达 2.9–4.8Å,全部 7 个 stem 形成(F1 0.89–0.95),但呈现**未被设计也未被 AF3 预测**的非规范三级相互作用。
- **中心思想**:不依赖 3D 结构预测、SS 中心+实验数据基础模型即可做 de novo RNA 设计。
- **与本项目零冲突**:全程**无配体条件下探测**(原文明言:假结形成要反映设计而非配体稳定;W03 c-di-GMP 核糖开关靶标挑战恰是"不靠配体也能折");所有参赛方法均不以靶标为条件。其 Discussion 明言 pocket 级高分辨率建模"对设计 RNA 催化剂与 **aptamer** 仍是一个重要挑战"——正是 TACo-Aptamer 的定位。
- 基线选择被背书:OpenKnot 场上最强的深度学习方法 = NA-MPNN 与 gRNAde(本项目重训基线)。

## 二、手稿改动(①②③ + 审计句)

1. **Intro**(第二段末):momentum 句——OpenKnot 刚在假结拓扑上追平人类(SHAPE/M2R/cryo-EM 验证),但全部设计无配体探测、无一方法以靶标为条件;"Topology-centric design has reached human parity; target-aware, binding-centred design remains unattempted."
2. **Discussion**(新增第二段):SS-only 的边界——OpenKnot 证明 SS 中心设计可成拓扑,但 (a) 目标是拓扑不是结合、无靶标条件;(b) cryo-EM 揭示非规范三级相互作用既非设计也非 AF3 可预测——SS 中心设计控制拓扑、留给口袋级三级几何未指定;对 aptamer 该几何就是规格,SS-only 与原子靶标条件化是互补而非替代。段末接泄露审计句(见 §四)。
3. **参考文献**:26 OpenKnot(Science 931–936)/ 27 Ribonanza-RNet(bioRxiv 2024.02.24.581671)/ 28 Struct2SeQ(bioRxiv 2026.01.16.700031)/ 29 Favor RFDdiff(bioRxiv 2025.10.01.679929)。

## 三、RNet 筛选实验设计(已否决,留档)

- **信号**:RibonanzaNet 输入设计序列(纯 AUGC)→ 预测逐核苷酸 2A3(SHAPE 类)反应性 → 对输入 SS(与 INF 同一参考)配对状态的 **AUROC**(反应性高⇒应未配对;OpenKnot 式无阈值版)。信号只依赖设计序列+参考 SS,与 logP/INF 并列为逐设计筛选信号。
- **验证链**(全部通过后才进入审计):
  - 权重:`roos23/RibonanzaNet`(HF 镜像原版 Kaggle `RibonanzaNet.pt`,45MB)→ 原版仓库 `tools/RibonanzaNet`(Shujun-He,MIT)Network.py **strict 加载 576 键全匹配**,11,296,898 参数;token A0/C1/G2/U3/pad4;原始输出通道 0=2A3、1=DMS(make_submission 列交换反推);官方推理含正反序列平均,已实现。
  - 合理性:10 条 test 原生序列上 AUROC(反应性⇒未配对)= 0.64–0.88,配对位均反应性 ~0.18 vs 未配对 ~0.35——模型确实在读出 SS。
  - 筛选脚本:`data/90_rnet_screen.py`(与 84 号同口径重算 random/logP/INF/RNET/cons2/cons3/oracle 七曲线;入口 assert 审计 PASS 才放行——现已因 FAIL 自动拦截)。

## 四、泄露审计(89 号,核心结论)

**方法**(`data/89_rnet_leakage_audit.py`):
- 对象:RMDB 全库 1,024 个 RDAT(CC0,GitHub DasLab/rmdb.github.io 五个 data-* release,共 12.5GB,存 `data/rmdb_audit/`)——RNet 训练数据(Eterna 化学映射)的上游来源;Kaggle 原始 train_data.csv 需登录,但所有已测量序列均出自 RMDB。
- 序列提取:两种 RDAT 载体全覆盖——BigLib 型 `ANNOTATION_DATA:…sequence:` 字段(含全部已测突变体)与经典 M2 型 `SEQUENCE` 野生型 + `mutation:G126C` 重建(**格式为"参考碱基+位置+新碱基"**,按 OFFSET 换算;首版解析反了已修)。共扫描 **4,525,354 条实测序列**(含重建突变体 8,868 条)。
- 原生序列:`data/splits_v2/seqs_all.json`(canonical,完整覆盖三层;⚠️ taco_s1_v2 缓存缺 indep 全部与 zs 的 2024+ 部分,首版用了缓存导致漏审,已改数据源)——去重后 422 条(test 255 / indep 130 / zs 37;test 中位长仅 43nt,多拷贝晶体链致 477→255)。
- 匹配:12-mer 精确共享预筛(2-bit 编码 numpy;**逐对计数**——12-mer 对 9k 原生并集的随机碰撞高,跨对聚合会爆候选)→ edlib 精验证三级口径:精确 / 全局相似 ≥0.8 / 双向 HW 嵌入 ≤10% 编辑;另报 ≥6 共享 12-mer 的片段嫌疑。
- 耗时 25 分钟(CPU)。产物:`survey/89_rnet_leakage_audit.json` + `89_audit.log`。

**结果:verdict = FAIL**

| 口径 | 命中 / 422 |
|---|---|
| 精确一致 | 0 |
| 全局相似 ≥0.8 | 11 |
| **嵌入(≤10% 编辑嵌于更长构建序列)** | **227** |
| 片段嫌疑(≥6 共享 12-mer,仅报备) | 284 |

**分层污染率(高相似+嵌入)**:**test 186/255 = 73%、zs 25/37 = 68%、indep 27/130 = 21%**。
典型例:test:1Y26_X(add 腺嘌呤核糖开关)对 `ADDRSW_1M7_*`/`ADD71_STD`(DasLab 对同一 RNA 的 M2/修饰干扰实测)相似度 0.96–0.99、共享 ~60 个 12-mer;4LX5/4TZX/5E54 等核糖开关同病。Eterna 库大量测过天然 RNA(带接头侧翼),RNet 训练分布**确实包含**本项目测试层原生序列的近拷贝。

**决策(用户规则:有泄露即不加)**:
- RNet 信号**不进入论文任何图表**;Fig6e 维持 84 号五曲线(INF/logP/共识/oracle/random)不动。
- Discussion 加一句方法论事实(可删):Eterna 化学映射库对 test 层原生序列 56% 近拷贝已被测量(test 73% vs 时间隔离层 21%)——既是时间隔离划分必要性的独立佐证,也警示实验数据基础模型的信号在结构基准上必须先做污染审计。
- **副产物(科学价值)**:indep(2024+)污染率仅 21% vs test 73%,正是时间隔离在起作用——可直接用作审稿人问"为什么要时间层"的弹药。

## 五、遗留与资源

- `data/rmdb_audit/`:12.5GB RDAT 全库 + `RibonanzaNet.pt` 权重 + 下载/解析脚本。磁盘余 8.1T,**保留**(审稿人复现审计需要;如需瘦身可删 RDAT 留 JSON)。
- `tools/RibonanzaNet/`:上游克隆(MIT);不影响主环境。
- 若未来想救活该实验,出路只有两条:①在 RNet 训练数据上做leave-one-family-out 重训(成本高);②把 RNet 信号只在 indep 未污染的 103 条链上评估(统计力弱)。当前均不建议。
- 手稿中 RNet 相关内容自检:Intro/Discussion 均只作为外部工作引用+审计事实,无任何依赖 RNet 信号的图表或数字。✅
