# RNA_tertiary_stru_IF

RNA 三级结构逆折叠（inverse folding）研究项目：AlignIF 复现与数据管线、RNA 设计方法基准评估、配体/适配体验证实验，以及论文手稿与图表。

## 代码与数据公开

- 代码：本仓库（[GitHub](https://github.com/SII-ShenggengLin/RNA_tertiary_stru_IF)），MIT 许可证。
- 论文复现精选数据与选定最终权重：[Zenodo](https://doi.org/10.5281/zenodo.24005533)，CC BY 4.0。
- `zenodo_deposit/`：数据包构建脚本、逐文件 SHA-256 清单生成逻辑、Zenodo 元数据与 API 上传/发布脚本。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `analysis/` | 编号研究文档：文献调研、AlignIF 复现笔记、方法架构设计、基准评估方案与结果（P0 系列、AptaDB 回顾性验证等）、写作交接 |
| `paper/` | 论文手稿（`manuscript.md`） |
| `figures/` | 论文图 fig1–fig5 及各自生成脚本 |
| `attic_v1_scripts_20260819/` | v1 版数据管线脚本存档（链拆分、聚类、划分、转换等） |
| `data/*.py`、`data/*.sh` | 当前数据管线、训练、推理与评测脚本 |
| `data/splits_v2/` | 精选数据划分定义（聚类结果、train/val/test 划分、序列汇总） |
| `tools/PACoAptamer/`、`tools/USalign/` | 无独立 git 仓库的小工具源码（随本仓库提交） |

## 不在 git 内的大文件

以下目录体积过大（合计 >135G），仅在服务器上维护，不纳入版本库；论文复现所需精选数据通过 Zenodo 发布：

- `data/`（除根目录脚本和 `splits_v2/` 外）：RCSB PDB/CIF 原始数据及衍生数据集（`rcsb_pdb/`、`rcsb_cif/`、`alignif/`、`canonical_v2/` 等）
- `models/`：各方法（AlignIF、ODesign、gRNAde、RhoDesign、NA-MPNN、PACoAptamer 等）的训练权重与 checkpoint
- `tmp/`：临时工作区

`tools/` 下带独立 git remote 的第三方仓库也不提交，需要时自行克隆：

| 本地路径 | 上游 |
| --- | --- |
| `tools/AlignIF/` | https://github.com/sfwang-auto/AlignIF.git |
| `tools/NA-MPNN/` | https://github.com/baker-laboratory/NA-MPNN.git |
| `tools/ODesign/` | https://github.com/OTeam-AI/ODesign.git |
| `tools/RhoDesign_official/` | https://github.com/ml4bio/RhoDesign.git |
| `tools/USalign-pylelab/` | https://github.com/pylelab/USalign.git |
| `tools/ccdutils/` | https://github.com/PDBeurope/ccdutils.git |
| `tools/gRNAde/` | https://github.com/chaitjo/geometric-rna-design.git |
| `tools/rnet-inference/` | https://github.com/DasLab/rnet-inference.git |
