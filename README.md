# PACo-Aptamer

PACo-Aptamer is a ligand-conditioned RNA inverse-folding model. This repository contains the final model, command-line tools and figure-reproduction code. The corresponding data and model checkpoints are available from [Zenodo](https://doi.org/10.5281/zenodo.24005531) under CC BY 4.0; source code is under the MIT License.

## Installation

```bash
git clone https://github.com/SII-ShenggengLin/RNA_tertiary_stru_IF.git
cd RNA_tertiary_stru_IF
conda env create -f environment.yml
conda activate paco-aptamer
```

The environment includes PyTorch, NumPy, SciPy, Matplotlib and ViennaRNA. No PyTorch Geometric or torch-scatter installation is required. A GPU is recommended for training and full benchmark evaluation; CPU inference is supported.

## Download the data

Download `PACoAptamer_reproduction_data_v2.tar.gz` from the Zenodo record, then run:

```bash
tar -xzf PACoAptamer_reproduction_data_v2.tar.gz
cp -a PACoAptamer_reproduction_data_v2/data .
cp -a PACoAptamer_reproduction_data_v2/checkpoints .
```

To verify every packaged file:

```bash
cd PACoAptamer_reproduction_data_v2
sha256sum -c MANIFEST.sha256
cd ..
```

The package contains only final splits, training features, results, figure assets and the two released PACo-Aptamer checkpoints. The two checkpoints are `checkpoints/paco_aptamer/ensemble_member_1.pth` and `ensemble_member_2.pth`.

## Design RNA sequences

An example RNA chain is included as `data/examples/8HZD.pdb`:

```bash
python scripts/design.py \
  --pdb data/examples/8HZD.pdb \
  --chain A \
  --output outputs/8HZD_A.fasta
```

For another RNA structure, replace `--pdb` and `--chain`. The chain can be omitted when the PDB contains exactly one supported RNA chain. The released model accepts up to 512 nucleotides.

By default, the command averages the two models' next-nucleotide probabilities and samples 100 sequences at each of temperatures 0.1, 0.5 and 1.0. Use `--samples-per-temperature` to change the pool size, `--device cuda:0` to select a GPU, or `--seed` to change the random seed. RNAfold supplies the secondary-structure input. Alternatively, pass `--secondary-structure` with a dot-bracket string of matching length.

The released inference protocol uses the RNA backbone and secondary structure without ligand tokens. Ligand context is used during stage-2 finetuning.

## Reproduce training

Stage 1 uses the 5,773 training chains for 60 epochs. Stage 2 uses the 768 RNA–ligand complexes for 8 epochs, mixes stage-1 chains into each epoch and selects checkpoints by validation recovery.

```bash
python scripts/train.py --stage s1 --name stage1_standard
python scripts/train.py --stage s1 --name stage1_ssdropout --ss-dropout 0.15

python scripts/train.py --stage s2 \
  --init outputs/checkpoints/stage1_standard.pth \
  --name ensemble_member_1
python scripts/train.py --stage s2 \
  --init outputs/checkpoints/stage1_ssdropout.pth \
  --name ensemble_member_2 --ss-dropout 0.15
```

Use `--device cuda:0` for GPU training and `--data-dir` when the data package is stored elsewhere. The default batch size is four chains and the learning rate is 0.0001.

## Reproduce evaluation

```bash
python scripts/evaluate.py \
  --split test \
  --device cuda:0 \
  --output outputs/benchmark_test.json
```

The default protocol samples 100 sequences at each temperature and reports best-of-pool and mean temperature-0.1 sequence recovery. Available splits are `train`, `val`, `test`, `independent_test` and `zero_shot_targets`.

For a quick pipeline check, add `--max-chains 2 --samples-per-temperature 2`. This check is not a substitute for the full benchmark.

## Regenerate figures

The figure scripts read the final results and assets from the downloaded data package:

```bash
python scripts/reproduce_figures.py fig2
python scripts/reproduce_figures.py
```

The first command regenerates main Figure 2; the second regenerates all main, Extended Data and Supplementary figures. Outputs are written to `figures/out/`. Use `--data-dir` for a different data location and the `PACO_OUTPUT_DIR` environment variable for a different output directory.

The package provides final comparison results for all methods; third-party baseline training pipelines and non-selected checkpoints are not redistributed.
