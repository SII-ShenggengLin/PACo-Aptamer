# Supplementary Information

# PACo-Aptamer: ligand-conditioned inverse folding for RNA aptamer design

[Author list placeholder]

> Supplementary Information draft v1.55 (2026-11-04), companion to manuscript v1.84. Numerical statements are traceable to the main text, the frozen analysis reports and the per-chain results archived with the project. Figure scripts and machine-readable tables are available in the released repository.
> v1.55: Supplementary citation formatting finalized for Nature Methods. In-text citations now use superscript Arabic numerals without brackets, S prefixes or ref. wording, with citations placed after existing punctuation where applicable. The independent Supplementary References list is numbered 1–26. No reference assignment, scientific content or reported statistic changed.
> v1.54: Supplementary References renumbered independently in order of first appearance, with all in-text citations updated. Entries now follow Nature Methods reference style, with standardized journal, conference, preprint, dataset and software formats. Non-bibliographic annotations were removed, DOI metadata were rechecked where available, and the AlignIF page range was added. No scientific content or reported statistic changed.
> v1.53: Supplementary Table notes compressed. Repeated repository pointers were consolidated into one section-level statement, redundant introductory sentences were removed, and statistical definitions, sample sizes and essential experimental conditions were retained as concise notes. Table contents and all reported values are unchanged from v1.52.
> v1.52: Supplementary Tables content compressed before sentence-by-sentence polishing, reducing the section by approximately 49%. Redundant screening rows, empty similarity ranges, superseded protocol values, duplicate baselines and secondary result blocks were removed. Table 3 now uses CCD identifiers with full chemical names in the repository, Table 6 focuses on LOFO families, Table 8 retains method-level summaries, and Table 11 removes repeated categories and descriptions. All retained numerical values were verified against v1.51, and no reported result changed.
> v1.51: Supplementary Figures refined sentence by sentence with the author after the v1.50 pass. Figure titles and captions now use more specific and direct descriptions of the data, recovery analyses, inference speed and conditioning inputs. The Other group, RDKit parsing of SMILES strings, sequence-logo pool sizes, random sequences with the same recovery and the ligand-token permutation condition are now stated explicitly. Related wording was synchronized in Supplementary Note 3 and Supplementary Tables 3 and 4. No reported statistic changed.
> v1.50: Supplementary Figures polishing completed with the author. Supplementary Figure 1 now describes data preparation and uses diagram rather than funnel. The exclusion of complexes sharing clusters with test chains and the origins of the 93 zero-shot chains are stated directly, and unnecessary hyphenated compounds were replaced throughout the captions. Vague titles and wording were simplified, averaging and timing descriptions were clarified, and Supplementary Figure 12 now describes the permuted pairing matrix and ligand-token control consistently with Supplementary Note 3. Related titles in the table of contents were updated, and the rest of the supplement was checked for the revised terminology. All numerical results are unchanged.
> v1.49: Supplementary Note 8 polishing completed with the author. The title now refers directly to baseline retraining and inference. Official training procedures and method-specific training-set sizes are stated directly, and the redundant common-subset sentence was removed. The RhoDesign training script, ligand context during finetuning, reporting of the t = 0.1 mean and the alignment and scoring rules are now described explicitly. Sampling settings were made more consistent, and MStA is expanded at its first occurrence. Related wording throughout the supplement was checked, with other uses of procedure and protocol retained in their appropriate technical contexts. All numerical results are unchanged.
> v1.48: Supplementary Note 7 polishing completed with the author. The title now refers directly to weight averaging and validation results. Training configurations, repeated runs and the timing of member selection are stated explicitly, and checkpoint origins replace the basin metaphor. The choice between ensemble sizes is now explicitly tied to validation results, and the weight averaging findings are described as consistent with linear mode connectivity theory. Related wording was updated in Supplementary Tables 9 and 11. The WMT 2016 paper on ensemble decoding is now cited rather than the separate back-translation paper. All numerical results are unchanged.
> v1.47: Supplementary Note 6 polishing completed with the author. The title and opening sentence now state the capabilities and limits of design based only on structure directly. The comparison with experienced human experts uses plainer wording, and the sequence overlap transition now identifies a specific risk of data leakage. The article in "the AI methods" was retained because the phrase refers specifically to the OpenKnot participants. Related wording throughout the supplement was checked, with other uses of scope and matched retained in their appropriate technical contexts. All numerical results are unchanged.
> v1.46: Supplementary Note 5 polishing completed with the author. The section and Supplementary Table 10 now describe sequence overlap with the RNet training data directly. RMDB is expanded at its first occurrence, and the internal M2 label is replaced by an explicit description of mutant reconstruction from reference sequences and mutation annotations. Native benchmark sequence records are distinguished from RMDB sequences, and the comparison is now correctly described as RMDB sequences against benchmark native sequences rather than PACo-Aptamer training or validation chains. The three alignment criteria, possible-fragment flags and within-layer deduplication are stated directly, short result fragments were merged, and the decision not to use RNet for design ranking is stated without nominalisations. Related wording was updated in Supplementary Note 6, Supplementary Figures 2 and 6 and Supplementary Table 1. All numerical results are unchanged.
> v1.45: Supplementary Note 4 polishing completed with the author. Database entries are now distinguished from filtered RNA–target interactions, and the merged analysis set is described explicitly as a dataset. The redundant RhoFold+ success count was removed. Random-sequence comparisons now name the sequences and structural matching directly, and database-specific results state the mean advantage directly. The affinity analysis now states the limited evidence for affinity ranking explicitly, and the four robustness checks are written as complete sentences without label-led fragments. Similar vague wording was revised in Supplementary Figures 1 and 10 and Supplementary Tables 4 and 5. The 0.20 SD comparison and the experimental-versus-predicted structure check now match the underlying analyses. Two one-sided Wilcoxon tests now use the stated 0.5 null, correcting the rank-test P from 0.999 to 0.174 and the sensitivity-test P from 0.0039 to 0.875. All other numerical results are unchanged.
> v1.44: Supplementary Note 3 polishing completed with the author. The section was retitled to describe analyses of conditioning inputs during training and inference, and vague diagnostics wording was replaced throughout. The secondary structure shuffle is now correctly described as a positional permutation of the pairing matrix rather than a sequence shuffle. Attention and pocket analyses now specify teacher-forced scoring, ligand-token order controls, the relevant recovery measures and the separate analysis of 130 chains from 12 families. The placebo direction was corrected, and P = 0.16 is now correctly identified as a one-sided Wilcoxon result. Inline citations throughout the supplement were standardized, with every supplementary reference cited and listed. All previously reported effect sizes, confidence intervals and P values are unchanged.
> v1.43: Supplementary Note 2 polishing completed with the author. The generic section introduction was removed. Zero-shot chain inclusion, controls without attributed ligands, CCD identifiers, SMILES parsing and the OXV exclusion are now stated explicitly, and CCD is expanded at its first occurrence in the supplement. Automated ligand assignments are distinguished from manual annotation, with the three mismatches and corrections stated unambiguously. LOFO families now include chemical names and selection criteria, individual family removal is distinguished from cumulative family coverage, and the family sizes of 15 chains for GAI and eight for flavins are stated explicitly. All previously reported numerical results are unchanged.
> v1.42: Supplementary Note 1 polishing completed with the author. The section now opens directly with the data quality findings, generic introductory and archival statements were removed, code and geometry repairs are described directly as fixes, and every paragraph begins with a complete sentence. Unnecessary hyphenation, bold lead-ins, semicolons and explanatory colons were removed, and the 2 × 2 comparison was rewritten explicitly as two coordinate assignments and two checkpoints. No reported statistic changed.
> v1.41: Explanatory colons in prose were replaced with complete sentences throughout the supplement. Structural colons in section headings, bibliographic titles, URLs, ratios and Markdown formatting were retained. No reported statistic changed.
> v1.40: Telegraphic paragraph lead-ins were removed throughout Supplementary Notes 1–8. Each paragraph now begins with a complete sentence. Figure and table caption titles were retained, and no scientific wording or reported statistic changed.
> v1.39: Semicolons were removed throughout the Notes, figure captions, tables and editorial notes. Independent clauses were split into sentences, and statistical parentheticals now use commas. Only semicolons separating deposited chemical components were retained. No reported statistic changed.
> v1.38: Supplementary Notes formatting normalized. Bold paragraph lead-ins were converted to plain body text, while figure and table caption titles and reference italics were retained. No scientific wording or reported statistic changed.
> v1.37: Supplementary Tables 1–11 revised to replace unnecessary hyphenated compounds and internal naming conventions with plain scientific descriptions. Table titles, condition labels and model variant names were simplified without changing their order, values or statistical interpretation.
> v1.36: Supplementary Figures 1–12 revised to replace unnecessary hyphenated compounds with plain phrases, particularly in titles, experimental conditions and averaging descriptions. Panel content, cross-references and all reported statistics are unchanged.
> v1.35: Supplementary Notes 1–8 polished to reduce first-person wording and unnecessary hyphenated compounds. In Note 1, two generic introductory sentences were removed, the title was simplified, and patches and defects were replaced by corrections and errors. Similar revisions were applied to experimental condition names and section headings throughout the Notes. Established technical terms and all reported statistics are unchanged.
> v1.34: Supplementary Tables 1–11 cleaned of local result paths, script references, internal split labels, run identifiers and code-style column names. The obsolete intermediate-file repair row was removed from Table 1, and model variants in Table 11 are now identified by their training or ensembling procedures. Public database identifiers and all reported scientific results are unchanged.
> v1.33: Supplementary Figures 1–12 cleaned of local result paths, script paths and internal workflow terminology. Figure content, panel lettering, cross-references and all reported statistics are unchanged.
> v1.32: Supplementary Notes 1–8 cleaned of internal dataset labels, local filenames and paths, code variables, checkpoint filenames and run-specific condition names. These were replaced with reader-facing descriptions of reference native sequences, residue mappings, chemical annotations, ligand-attribution rules, perturbation controls and training procedures. Public database identifiers, software versions and all numerical results are unchanged.
> v1.31: Overall concision pass before sentence-by-sentence polishing, reducing the word count by approximately 43%. Internal revision history, repeated numerical summaries and duplicated provenance notes were removed. The Notes, figure captions and table preambles were streamlined, and Supplementary Table 3 was grouped by ligand annotation to eliminate repeated chemical names. All section, figure and table numbers and all reported statistics are unchanged.

## Table of Contents

**Supplementary Notes**

- Supplementary Note 1 — Data quality review and corrections to baseline code
- Supplementary Note 2 — Zero-shot ligand identities and ligand family definitions
- Supplementary Note 3 — Analysis of conditioning inputs during training and inference
- Supplementary Note 4 — Analysis of measured aptamer–target interactions
- Supplementary Note 5 — Screen for sequence overlap with the RNet training data
- Supplementary Note 6 — The OpenKnot challenge and the limits of design based only on structure
- Supplementary Note 7 — Ensemble selection, weight averaging and validation results
- Supplementary Note 8 — Baseline retraining and inference

**Supplementary Figures**

- Supplementary Figure 1 — Data preparation and the three test layers
- Supplementary Figure 2 — Dataset contents and data splits
- Supplementary Figure 3 — Design confidence and sequence recovery across and within chains
- Supplementary Figure 4 — Perplexity of native sequences
- Supplementary Figure 5 — Recovery by ligand family in the cluster-based test set
- Supplementary Figure 6 — PACo-Aptamer's recovery advantage over RhoDesign decreases as test ligands become more similar to training ligands
- Supplementary Figure 7 — Predicted backbones reduce sequence recovery
- Supplementary Figure 8 — PACo-Aptamer generates sequences faster than the baseline methods
- Supplementary Figure 9 — Interface sequence logos and agreement with native sequences
- Supplementary Figure 10 — Self-consistency metrics for 193 chains with predicted structures
- Supplementary Figure 11 — Sampling temperature trades recovery for diversity
- Supplementary Figure 12 — Analysis of conditioning inputs

**Supplementary Tables**

- Supplementary Table 1 — Data quality screening of entries removed, repaired and reviewed
- Supplementary Table 2 — Sequence recovery in the three test layers
- Supplementary Table 3 — Zero-shot ligand identities
- Supplementary Table 4 — Ligand similarity and recovery advantage, all fingerprints
- Supplementary Table 5 — Leave one family out and family coverage results
- Supplementary Table 6 — Ligand family definitions for the LOFO analysis
- Supplementary Table 7 — Benchmarks using predicted backbones, by method
- Supplementary Table 8 — Inference cost per method
- Supplementary Table 9 — Ablations of conditioning inputs
- Supplementary Table 10 — Sequence overlap with the RNet training data by criterion and evaluation layer
- Supplementary Table 11 — Evaluation of PACo-Aptamer training and ensemble variants

**Supplementary References**

---

## Supplementary Notes

### Supplementary Note 1: Data quality review and corrections to baseline code

Forty crystal chains were stored as tandem concatenations of the same molecule, including three chains in the cluster-based test set. For example, 1EI2_A contained 17 copies of a 25-nt chain and 2KGP_A contained 10. The 104 chains associated with the finetuning pool were rebuilt as all-atom PDBs containing one copy of each molecule, and the reference set of 6,346 native sequences was regenerated using the same procedure. All evaluations use these native sequences with one molecular copy. The earlier stage 1 pretraining run included approximately 37 concatenated sequences (at most 1.2% of its training data). The model was not retrained because these sequences affected the training distribution rather than the test references.

Twenty-nine chains labelled as RNA were traces of protein backbones. All occurred in the training split and were removed from baseline training. Modified nucleotides, the ambiguous residue N in the PDB Chemical Component Dictionary (CCD) and 5′ variants such as +U were mapped to their parent bases or marked undetermined using a curated mapping of residue names. Legacy atom names were normalised in 13 chains (4 test, 9 training). Positions with undetermined native bases are excluded from every denominator.

Reproducing RhoDesign, gRNAde, NA-MPNN and AlignIF<sup>1–4</sup> required fixes to the source code, all available in the released repository. For AlignIF, FASTA output across gaps in PDB numbering, mapping of residue names with multiple characters and Boolean parsing in the command line interface were fixed. Inference was then rerun in full. For NA-MPNN, handling of missing C1′ coordinates, insertion codes and uncommon residue names was fixed, and graceful fallbacks were added for missing optional dependencies. Five test chains lacking C1′ coordinates in their source PDBs (5NFV_B, 7E9I_A, 8U5J_A, 8VPV_A, 9GUT_X) are scored on valid positions only. For RhoDesign, the NMR entries 2KGP_A and 6IZP_A were rerun using reconstructed reference inputs containing one molecular copy instead of concatenations of ten models.

The original inference workflow wrote sampled tokens in A,C,G,U order, whereas gRNAde training uses A,G,C,U, thereby interchanging C and G during decoding. Scoring of native sequences and a paired comparison of 40 chains confirmed the official order. Because the two outputs are related by a deterministic C/G interchange, the output pools were reconstructed exactly using the official nucleotide order. On the cluster-based test set, this fix raised gRNAde recovery from 50.15% to 53.00% for the best sample and from 31.21% to 35.77% for the t = 0.1 mean. All downstream metrics were recomputed, and every PACo-Aptamer advantage over gRNAde remained significant (all P ≤ 2.9 × 10⁻¹⁷).

The representation of nine atoms used for likelihood scoring (the geometric vector perceptron input)<sup>5</sup> had placed the glycosidic N9 atom of purines in the coordinate slot defined during training as N1. This removed the geometric distinction between purines and pyrimidines. A comparison of two coordinate assignments and two checkpoints on 40 AptaDB chains isolated the error. N1 geometry gave advantages of +2.22 and +2.11 log units for the two checkpoints, whereas N9 geometry gave −0.42 and −0.45. All interaction analyses use N1 geometry after the fix.

### Supplementary Note 2: Zero-shot ligand identities and ligand family definitions

The zero-shot layer contains 93 chains (54 cluster-based and 39 time-based) whose attributed ligands, when present, share no CCD codes with any ligand in the training or finetuning data. Of these, 88 chains have at least one attributed ligand. These ligands represent 43 distinct CCD codes. The other five (5NFV_B, 6IV6_G, 6VRD_B, 8SWB_B, 8SWC_B) are retained as controls without an attributed ligand because their deposited structures contain no functional ligand. Supplementary Table 3 lists the ligand identities, chain counts and representative chains.

The ligands attributed to the 768 finetuning chains from 510 PDB entries correspond to 247 distinct CCD codes. Canonical CACTVS SMILES strings in the PDB Chemical Component Dictionary were used when available, with other deposited SMILES strings used otherwise. The SMILES strings for five training ligands (6NO, 6O1, B12, B1Z, HEM) cannot be parsed by RDKit,<sup>6</sup> leaving 242 training ligands with usable Morgan fingerprints.<sup>7</sup>

The similarity analysis includes 87 of the 88 attributed zero-shot chains, with 6PQ7_C excluded because its ligand OXV has no SMILES string that can be parsed by RDKit. The median across chains of the maximum Tanimoto similarity to any training ligand is 0.379. One of 87 chains reaches similarity ≥ 0.9, with a maximum of 1.0.

Ligands were assigned at PDB level by an ordered cascade comprising pairing based on deposited annotations (383 of 510 entries), union of symmetry-related copies (23), geometric pairing at ≥ 0.8 contact overlap (8), geometric pairing requiring at least 4 contacts involving heavy atoms (34) and relaxed geometric pairing (62). Automated ligand assignments matched manual annotation for 112 of 115 chains (97.4%). The three mismatches arose from incomplete deposition of multiple molecular copies, with QUA reassigned to NO1 in 1OLN and 5D8H and C5P reassigned to XY7 in 8GS2.

Supplementary Table 6 lists the CCD codes included in each ligand family. The leave one family out (LOFO) analysis uses the five largest ligand families, namely aminoglycosides, S-adenosylmethionine and S-adenosylhomocysteine (SAM/SAH), chlorophyll a (CLA), nucleotide triphosphates (NTP) and cyclic di-AMP (2BA). The sixth-largest family, guanidine (GAI), has 15 chains and is not included in the rotation. Flavins are excluded because that family contains only eight chains. Removing each LOFO family in turn leaves 640, 713, 744, 748 and 752 training chains, respectively. The cumulative coverage pools contain 128, 183, 207 and 243 chains for one, two, three and all five families, respectively, compared with 768 chains in the full training pool.

### Supplementary Note 3: Analysis of conditioning inputs during training and inference

These analyses tested whether the performance advantage could be attributed to secondary structure, ligand context<sup>8</sup> and interface contacts rather than evaluation confounds. Unless stated otherwise, results use the published ensemble of two models, 300 samples per chain and the cluster-based test set of 477 chains.

Recomputed scoring reproduced the previously generated inference pools within numerical tolerance on 469 of 477 chains, and the remaining eight showed a mean absolute change of 0.019 in recovery of the best sample. The recomputed mean recoveries across chains (0.7464 for the best sample and 0.5970 at t = 0.1) fall within the bootstrap confidence intervals of the reported benchmark values (0.749 and 0.604). The following analyses therefore use the recomputed scoring workflow.

The standard secondary structure input was compared with three altered versions. One had all nucleotides unpaired, one used the same positional permutation on the rows and columns of the pairing matrix to disrupt the pairing pattern while preserving the number of pairs, and one used the centroid of the RNAfold ensemble rather than the structure with the lowest free energy (ViennaRNA).<sup>9</sup> Relative to the standard input, the input with all nucleotides unpaired changes recovery of the best sample by −0.28 percentage points (95% CI −0.55 to 0.00) and t = 0.1 mean recovery by −0.46 points (−0.62 to −0.30). The corresponding changes are −0.26 (−0.57 to +0.05) and −0.59 (−0.77 to −0.43) for the permuted pairing matrix, and +0.17 (−0.09 to +0.44) and +0.002 (−0.08 to +0.08) for the input derived from the ensemble centroid. Thus, altering the secondary structure input at inference changes recovery by less than 0.6 percentage points. Removing secondary structure during training has larger effects (Supplementary Table 9), suggesting that it guides model training but adds little information beyond backbone geometry at inference.

For 115 chains with interface annotations (905 interface positions), decoder cross-attention weights assigned to ligand tokens were recorded during teacher-forced scoring of the native sequence. Across chains, the mean interface-minus-non-interface difference in attention assigned to ligand tokens is −0.0003 of the total attention per position (95% CI −0.0020 to +0.0013, one-sided Wilcoxon P = 0.879, with higher interface attention in 31.3% of chains). The corresponding mean differences are −0.0003 (P = 0.888) for the published ensemble with permuted ligand tokens and −0.0018 (P = 0.99994) for the single standard model. The mean Spearman correlation across chains between attention assigned to ligand tokens and the distance from each nucleotide C1′ atom to the nearest ligand atom is +0.104 (median 0.257, 33% of chains negative). Decoder cross-attention to ligand tokens therefore does not systematically increase at interface positions or at nucleotides closer to the ligand.

On the same 115 chains, true and empty pocket inputs were compared in the standard single model using pools of 30 samples. The true pocket changes interface recovery of the best sample by +1.04 percentage points relative to the empty pocket (95% CI −1.1 to +3.1, one-sided Wilcoxon P = 0.16). In a placebo control, feeding ligand tokens to a variant trained without ligand conditioning lowers overall recovery of the best sample by 15.6 percentage points. In a separate analysis of 130 chains from 12 families, poses of the same ligand were transplanted from the nearest alternative structure within each family by C1′ Kabsch superposition to test whether pocket-placement noise obscured a larger effect (median centroid displacement 0.85 Å, 90th percentile 2.25 Å). The mean difference in log probability per interface position between true and transplanted pockets was approximately +0.0002 nats and was not significant.

The ablations in Supplementary Table 9 and the analyses above indicate that conditioning inputs contribute during training, whereas the published model relies on backbone geometry and sequence context rather than on any single conditioning input at inference.

### Supplementary Note 4: Analysis of measured aptamer–target interactions

Screening 1,350 AptaDB entries<sup>10</sup> and 1,495 UTexas entries<sup>11</sup> using the Methods filters for measured affinity, sequence validity and target identity yielded 254 AptaDB and 228 UTexas RNA–target interactions. To exclude leakage from training, all aptamers were clustered with training chains using CD-HIT-EST<sup>12</sup> at 0.8 sequence identity, and every aptamer sharing a cluster with a training sequence was removed. Forty AptaDB interactions were removed, including three exact training duplicates, leaving 214 clean interactions. No UTexas interactions were removed. The merged clean dataset contains 442 interactions, and the sensitivity analysis uses all 254 AptaDB interactions.

Each aptamer was folded with RhoFold+<sup>13</sup> and scored by teacher forcing with the published ensemble. For the primary comparison, ten random sequences of the same length were scored against the aptamer's predicted structure and secondary structure. In a second comparison, each random sequence was instead assigned its own secondary structure. The primary statistic is the aptamer's log probability minus the mean log probability of its structure-matched random sequences.

The merged clean dataset has a mean advantage of +1.95 log units, and all 442 aptamers score above the mean of their structure-matched random sequences. For AptaDB, the mean advantage is +2.01 (95% CI 1.94 to 2.09, n = 214), with +2.03 for the full sensitivity set of 254 interactions. For UTexas, the mean advantage is +1.90 (1.80 to 2.00, n = 228).

Across 19 families with at least three measured Kd values, the mean within-family Spearman correlation between model likelihood and affinity is ρ = 0.11 (95% CI −0.15 to 0.36, median 0.22), and 58% of the family-level correlations are positive. The sensitivity analysis of 38 families with at least two measurements shows a similarly weak association. For the highest-affinity aptamer in each family, the mean percentile rank under the model is 0.439, with 0 denoting the highest model score (95% CI 0.28 to 0.61, one-sided Wilcoxon P = 0.174 for ranks below 0.5). After within-family z-score standardisation and pooling of 84 aptamer–target pairs, the Spearman correlation is ρ = 0.046. Aptamers in the top model-likelihood quartile have a mean standardised affinity 0.20 SD higher than that of the remaining aptamers. In an exploratory analysis of 10 families, the mean sensitivity for detecting the highest-affinity half among the model's top-ranked half is 0.433 (95% CI 0.267 to 0.600, one-sided Wilcoxon P = 0.875 for sensitivity above 0.5). Overall, these analyses provide no consistent evidence that model likelihood reliably ranks aptamers by affinity.

Scores under experimental structures exceed those under predicted structures by only 0.003 log units per nucleotide after the scoring repair, compared with 0.054 before the repair. Scoring aptamers against unrelated structures reduces the advantage, indicating that the advantage depends on structure rather than sequence composition alone. After the repair, GC content correlates with likelihood at ρ = 0.02, compared with −0.116 before the repair. Excluding annotated G-quadruplex aptamers does not change these conclusions.

### Supplementary Note 5: Screen for sequence overlap with the RNet training data

RNet helped filter designs in the OpenKnot pseudoknot challenge<sup>14,15</sup> and was trained on chemical mapping data from the RNA Mapping Database (RMDB).<sup>16</sup>

The source data comprised 1,024 RDAT files from five RMDB releases.<sup>16</sup> These files provided 4,525,354 sequences, including 8,868 mutant sequences reconstructed from reference sequences and mutation annotations such as G126C, which denotes replacement of G at position 126 by C. After duplicate sequences were removed within each evaluation layer, 422 native benchmark sequence records remained, comprising 255 test, 130 independent and 37 zero-shot records.

Candidate pairs of RMDB and benchmark sequences were identified by at least two shared 12-mers and aligned with the Edlib sequence alignment library.<sup>17</sup> Three criteria were used to define sequence overlap. These were an exact match after U/T normalisation, global edit-distance similarity ≥ 0.8, and semi-global alignment showing that either sequence was embedded in the other with edit distance ≤ 10% of the embedded sequence length. Benchmark sequences that shared at least six 12-mers with an RMDB sequence but had no alignment meeting these criteria were recorded separately as possible fragments and were not counted as overlaps.

After the three criteria were combined, 227 of the 422 native sequences were flagged (53.8%), comprising 178 of 255 test (69.8%), 27 of 130 independent (20.8%) and 22 of 37 zero-shot sequences (59.5%). No exact matches were found, and all 11 sequences with high global similarity also met the embedding criterion. Supplementary Table 10 reports both the deduplicated counts and the counts summed across criteria. This overlap reflects the origin of the chemical mapping data, which include extensively studied RNAs that are also represented in the PDB, rather than the benchmark split.

No benchmark subset free of overlap with the RNet training data was large enough for reliable evaluation. RNet was therefore not used to rank designed sequences in any reported benchmark, including those in the main text figures and tables. Screening results for each benchmark sequence are available in the released repository.

### Supplementary Note 6: The OpenKnot challenge and the limits of design based only on structure

The OpenKnot challenge<sup>15</sup> clarifies the capabilities and limits of design based only on structure. Deep learning methods performed as well as experienced human experts on de novo pseudoknot targets without 3D structure prediction, guided by RNet,<sup>14</sup> which was trained on roughly two million chemical mapping measurements from Eterna. However, OpenKnot tested every design without small-molecule ligands and targeted correct secondary structure rather than binding. No participating framework conditioned on a target molecule. Moreover, cryo-EM revealed noncanonical tertiary interactions that were neither designed by the AI methods nor predicted by AlphaFold 3.<sup>18</sup>

Design centred on secondary structure can therefore control topology while leaving the tertiary pocket geometry required for binding unspecified. For aptamers, secondary structure design and atomic target conditioning are complements rather than substitutes. The screen for sequence overlap reveals a separate risk of data leakage. Nearly identical copies of many benchmark native sequences had already been measured in the libraries used to train RNet, with markedly different rates in the cluster-based and time-based layers (Supplementary Note 5 and Supplementary Table 10). This overlap cannot be detected by checking only the training split derived from structural databases and motivates both a time-based holdout and explicit screening of external training datasets.

### Supplementary Note 7: Ensemble selection, weight averaging and validation results

The released model is an ensemble of two stage 2 checkpoints, each selected on validation data. Development variants changed the random seed, secondary structure dropout, batch size or exponential moving average, or repeated the base training run without changing its configuration. Apart from the specified changes, stage 2 training was held constant, and all test evaluations used pools of 300 samples. In the development scan, the standard variant (the base training configuration with training seed 43) and the variant with secondary structure dropout of 0.15 had the highest recovery of the best sample (71.49% and 71.45%, respectively). All other variants considered before selection reached at most 70.95%. The two selected models differ in training seed and regularisation and were finetuned from different stage 1 checkpoints. A later run combining seed 43 with secondary structure dropout of 0.15 reached 71.44%, but it was performed only after the two ensemble members had been selected. Training and ensemble decoding use fairseq.<sup>19</sup> Supplementary Table 11 gives the full results and describes each variant.

The ensemble averages the two members' stepwise softmax distributions before multinomial sampling. The pool size is unchanged, and inference cost doubles. A three-member ensemble gave slightly higher recovery of the best sample on the cluster-based test set (75.2% against 74.6% under the evaluation convention used during development) but did not improve validation recovery (best 0.8256 against 0.8263, t = 0.1 mean 0.7256 against 0.7331). The two-member ensemble was therefore retained, and these test scores were not used to choose between the two- and three-member ensembles. After interface weighting was removed, the two-member ensemble and its individual members were evaluated again on the 96 validation chains, and the two-member ensemble was retained. The best single model still outperforms every baseline on every test layer, so no main text claim depends on ensembling.

Averaging the weights of two to four checkpoints gave 66.17–70.90% recovery of the best sample, compared with 71.5% for the best single model. Averaging two stage 2 checkpoints derived from the same stage 1 checkpoint did not improve recovery, whereas averaging checkpoints derived from different stage 1 checkpoints reduced recovery. This pattern is consistent with linear mode connectivity theory. Weight averaging can fail when checkpoints are not linearly connected in weight space, whereas averaging output probabilities can benefit from diversity among independently trained models.<sup>20–23</sup>

### Supplementary Note 8: Baseline retraining and inference

RhoDesign, gRNAde, NA-MPNN and AlignIF were retrained using their official training procedures and the same split of 5,773 training chains and 96 validation chains. Their official featurization pipelines retain different subsets, so the number of training chains ranges from 5,584 to 5,773 across methods (Extended Data Table 1).

For RhoDesign, a training script written for this study used the official model code, and the checkpoint was selected by validation recovery. gRNAde used the official training script and checkpoint-selection procedure. NA-MPNN used official hyperparameters (hidden dimension 128, 3 + 3 encoder/decoder layers, k = 32 neighbours, Noam schedule, label smoothing 0.1, automatic mixed precision, batches of 6,000 tokens and 20,000 optimization steps). All 21 saved checkpoints were evaluated, and the checkpoint at step 11,016 was selected (epoch 153, validation top-1 accuracy 0.6771). AlignIF completed its official schedule of 80 epochs, with validation loss decreasing from 0.880 to 0.697. The final checkpoint (epoch 79) was used.

During benchmark inference, every method conditions only on the backbone and predicted secondary structure. PACo-Aptamer uses ligand context during finetuning but not at inference. PACo-Aptamer, RhoDesign and gRNAde each sample 100 sequences at each of three temperatures (0.1, 0.5 and 1.0). RhoDesign retains its official downsampling of long chains (50 sequences per temperature above 200 nt and 30 above 400 nt). AlignIF samples 20 sequences at t = 0.1 without retrieval of multiple structure alignments (MStA), and NA-MPNN samples 10 sequences at t = 0.1. Comparisons based on the best sample mildly disadvantage methods with smaller pools, so the t = 0.1 mean is also reported. Outputs shorter than the native sequence are aligned by matching equal blocks and scored on valid positions only, excluding undetermined native bases. The same alignment and scoring rules apply to evaluations with predicted backbones and efficiency measurements.

## Supplementary Figures

**Supplementary Figure 1. Data preparation and the three test layers.** The diagram shows how the data were prepared, split and assigned to the three test layers. Complexes sharing a cluster with any test chain were excluded, leaving 768 chains from 510 PDB entries for stage 2 finetuning. The zero-shot test layer contains 93 chains, including 54 from the cluster-based test set and 39 from the time-based test set. Their attributed ligands, when present, are absent from training complexes. No test chain is included in training or finetuning.

**Supplementary Figure 2. Dataset contents and data splits.** (A) Chain counts for the training, validation and finetuning data and the three test layers. The zero-shot chains also belong to the cluster-based or time-based test sets. (B) Distributions of chain length across data splits. (C) Ligand families in the cluster-based test set. Of the 477 chains, 105 belong to nine annotated ligand families and the remaining 372 to Other. The interface analysis includes 115 chains and 905 interface positions.

**Supplementary Figure 3. Design confidence and sequence recovery across and within chains.** (A) Across-chain and (B) within-chain Spearman correlations between design log probability and native recovery. NA-MPNN pools contain only 10 samples and are shown for descriptive comparison. (C) Recovery after selecting the top 10% of designs by log probability. gRNAde pools use corrected C/G decoding and scoring with the official vocabulary. Supplementary Note 3 describes the recomputed scoring workflow and the eight affected chains.

**Supplementary Figure 4. Perplexity of native sequences.** (A) Perplexity of native sequences for the 477 chains in the cluster-based test set. Black lines mark the mean perplexity for each method. AlignIF is excluded because it cannot score native sequences. (B) Paired perplexity differences for each chain (baseline − PACo-Aptamer).

**Supplementary Figure 5. Recovery by ligand family in the cluster-based test set.** Mean recovery at t = 0.1 for five methods across nine annotated ligand families. The Other group contains the 372 chains outside these nine families. PACo-Aptamer outperforms RhoDesign in eight families and in Other. Mango-III is the only exception. Full recovery values and paired tests are in Extended Data Table 3.

**Supplementary Figure 6. PACo-Aptamer's recovery advantage over RhoDesign decreases as test ligands become more similar to training ligands.** Each point represents one of 87 zero-shot chains whose ligands have SMILES strings that RDKit can parse. The y axis shows the recovery advantage of PACo-Aptamer over RhoDesign (t = 0.1 mean), and the x axis shows the maximum Tanimoto similarity to training ligands. Similarity is calculated using (A) Morgan, (B) MACCS<sup>24</sup> and (C) RDKit topological fingerprints. Each panel shows a least-squares fit and a zero line marking equal recovery for the two methods. Supplementary Table 4 provides correlations for both recovery measures and results by similarity range.

**Supplementary Figure 7. Predicted backbones reduce sequence recovery.** (A) Recovery of the best sample using crystal backbones and backbones predicted by RhoFold+ for the same 193 chains. Changes in the t = 0.1 mean are also annotated. (B) For the 32 zero-shot chains, the PACo-Aptamer advantage over RhoDesign remains significant for the best sample but not for the t = 0.1 mean. (C) Predicted backbones reduce recovery for both methods in this subset. Supplementary Table 7 provides absolute values and paired tests.

**Supplementary Figure 8. PACo-Aptamer generates sequences faster than the baseline methods.** (A) Runtime per chain, (B) time per sample, (C) throughput and (D) peak GPU memory for six probe chains per method. The y axes in B and C use log scales. Bars in A and D show means, and error bars show the minimum and maximum values. PACo-Aptamer uses the released ensemble of two models and pools of 300 samples. Supplementary Table 8 provides all values. Comparisons across methods should be interpreted with caution because probe sets, backbone inputs, GPUs and conventions for measuring memory differ.

**Supplementary Figure 9. Interface sequence logos and agreement with native sequences.** (A, B) Interface sequence logos for 8HB1_A and 8D5L_A. Letter height shows nucleotide frequency in each method's t = 0.1 pool, and the native row shows the corresponding native sequence. Pools contain 100 sequences for PACo-Aptamer, RhoDesign and gRNAde, 20 for AlignIF and 10 for NA-MPNN. (C) Agreement with native sequences at 905 interface positions across 115 chains. For each method and chain, the most frequent nucleotide in the design pool is compared with the native nucleotide. For each method, the first value is the mean across all positions, and the second is the mean of the values calculated separately for each chain.

**Supplementary Figure 10. Self-consistency metrics for 193 chains with predicted structures.** (A) scMCC against scTM for each chain, with crosses marking method means. scMCC compares the base pairs identified by barnaba in each refolded structure with the native base pairs.<sup>25</sup> scTM measures structural similarity to the native structure after RhoFold+ refolding using US-align.<sup>26</sup> (B) Mean values for native and designed sequences and for random sequences with the same recovery as the PACo-Aptamer designs. The analysis includes 193 chains, with valid metrics for 160–174 chains per condition and 157–162 chains shared between conditions for paired tests. The two metrics are complementary, with Spearman correlations across chains of ρ = 0.18–0.61.

**Supplementary Figure 11. Sampling temperature trades recovery for diversity.** Mean recovery, pairwise diversity and the fraction of unique sequences at t = 0.1, 0.5 and 1.0 for PACo-Aptamer (477 chains), RhoDesign (437) and gRNAde with corrected decoding (437). PACo-Aptamer retains higher recovery than RhoDesign at similar levels of diversity and yields 88.2% unique sequences at t = 0.1. Recovery uses the recomputed scoring workflow in Supplementary Note 3.

**Supplementary Figure 12. Analysis of conditioning inputs.** (A) Changes in recovery of the best sample and in mean recovery at t = 0.1 when the standard secondary structure input is replaced by an input with all nucleotides unpaired, a positionally permuted pairing matrix or the ensemble centroid (477 chains). Error bars show bootstrap 95% CIs. No recovery change exceeds 0.6 percentage points in either direction. (B) Difference in decoder cross-attention to ligand tokens between interface and non-interface positions for the published ensemble, the same ensemble with permuted ligand tokens and the single standard model (115 chains). Attention differences are expressed as percentage points of total attention per position. Supplementary Note 3 provides exact values and the attention–distance and pocket analyses.

## Supplementary Tables

**Supplementary Table 1. Data quality screening of entries removed, repaired and reviewed.**

| Category | Scope | Affected chains | Correction |
|---|---|---|---|
| Crystal chains with concatenated repeated copies | Finetuning and evaluation | 40, including 3 test chains, and about 37 stage 1 training chains (≤1.2%) | One molecular copy restored and native sequences regenerated. Stage 1 was not retrained |
| Protein backbone traces | Training | 29 | Removed from baseline training |
| Residue and atom name normalisation | All chains | 684 CCD mappings and legacy O5* names in 13 chains (4 test, 9 training) | Modified residues mapped to parent bases. Undetermined positions excluded from scoring |
| Missing C1′ coordinates | NA-MPNN test inputs | 5 (5NFV_B, 7E9I_A, 8U5J_A, 8VPV_A and 9GUT_X) | Scored on valid positions only |

**Supplementary Table 2. Sequence recovery in the three test layers**

| Layer | n | Method | Best sample (%) | t = 0.1 mean (%) | Δ best (pp) | P best | Δ t = 0.1 (pp) | P t = 0.1 |
|---|---|---|---|---|---|---|---|---|
| cluster-based test | 477 | PACo-Aptamer | 74.9 | 60.4 | — | — | — | — |
| cluster-based test | 477 | RhoDesign | 69.4 | 55.1 | +5.5 | 4.8e-48 | +5.3 | 6.1e-32 |
| cluster-based test | 477 | gRNAde | 53.0 | 35.8 | +21.9 | 1.5e-79 | +24.6 | 1.2e-79 |
| cluster-based test | 477 | NA-MPNN | 52.2 | 47.0 | +22.7 | 1.3e-77 | +13.4 | 1.0e-62 |
| cluster-based test | 477 | AlignIF | 56.2 | 51.8 | +18.7 | 4.0e-77 | +8.6 | 2.0e-42 |
| time-based independent test | 235 | PACo-Aptamer | 74.4 | 58.3 | — | — | — | — |
| time-based independent test | 235 | RhoDesign | 71.3 | 56.3 | +3.2 | 2.7e-12 | +1.9 | 5.1e-5 |
| time-based independent test | 235 | gRNAde | 51.3 | 32.4 | +23.2 | 4.5e-40 | +25.9 | 4.0e-40 |
| time-based independent test | 235 | NA-MPNN | 45.6 | 39.5 | +28.8 | 5.6e-40 | +18.8 | 4.5e-39 |
| time-based independent test | 235 | AlignIF | 51.0 | 45.6 | +23.4 | 1.1e-39 | +12.6 | 2.8e-30 |
| zero-shot test (ligand holdout) | 93 | PACo-Aptamer | 76.1 | 62.8 | — | — | — | — |
| zero-shot test (ligand holdout) | 93 | RhoDesign | 71.8 | 59.5 | +4.3 | 7.2e-9 | +3.4 | 4.7e-5 |
| zero-shot test (ligand holdout) | 93 | gRNAde | 55.8 | 39.2 | +20.3 | 5.6e-17 | +23.7 | 5.8e-17 |
| zero-shot test (ligand holdout) | 93 | NA-MPNN | 57.1 | 52.5 | +19.0 | 2.4e-16 | +10.3 | 3.0e-11 |
| zero-shot test (ligand holdout) | 93 | AlignIF | 59.5 | 56.2 | +16.6 | 5.7e-16 | +6.6 | 2.9e-7 |

Values are means across chains. Δ = PACo-Aptamer − comparator in percentage points. P values are two-sided paired Wilcoxon signed-rank tests, excluding pairs with zero difference.

**Supplementary Table 3. Zero-shot ligand identities**

The 93 zero-shot chains are grouped by CCD code or deposited code combination.

| CCD code | Chains | Max Tanimoto |
|---|---:|---:|
| MMC | 3 | 0.200 |
| MIX | 1 | 0.176 |
| PQ0 | 1 | 0.564 |
| 5AZ | 1 | 0.324 |
| DGP | 2 | 0.847 |
| 29G | 1 | 0.200 |
| 29H | 1 | 0.385 |
| n/a | 5 | n/a |
| RIV | 1 | 0.472 |
| HLV | 2 | 0.179 |
| HMJ | 1 | 0.222 |
| HMV | 1 | 0.230 |
| HNG | 1 | 0.579 |
| HZG | 4 | 0.379 |
| HZD | 1 | 0.653 |
| B2R | 1 | 0.206 |
| OXV | 1 | n/a |
| YO3 | 2 | 0.373 |
| UG1 | 2 | 0.316 |
| UG4 | 2 | 0.300 |
| J0C | 2 | 0.322 |
| XAN | 6 | 0.441 |
| AZA | 2 | 0.231 |
| KY6 | 1 | 0.696 |
| NPR | 2 | 0.275 |
| QB3 | 2 | 0.333 |
| QAX | 2 | 0.295 |
| QIJ | 4 | 0.267 |
| QEU | 2 | 0.311 |
| O2I | 2 | 0.558 |
| NI4 | 6 | 0.328 |
| NJL | 2 | 0.617 |
| O00 | 2 | 0.617 |
| V7U+V8C | 3 | 0.729 |
| P5E | 1 | 0.833 |
| N2P | 1 | 1.000 |
| 141 | 1 | 0.282 |
| AR6 | 10 | 0.875 |
| W6F | 1 | 0.465 |
| DHF | 1 | 0.213 |
| AZG | 1 | 0.163 |
| ANG | 1 | 0.567 |
| AR6+NCA | 4 | 0.875 |

Max Tanimoto uses Morgan fingerprints (radius 2, 2048 bits) against 247 training ligands. Five chains have no attributed ligand, and RDKit cannot parse the OXV SMILES string, leaving 87 chains for similarity analysis.

**Supplementary Table 4. Ligand similarity and recovery advantage, all fingerprints**

| Fingerprint | Range | n | Δ best (pp) | P best | Δ t = 0.1 (pp) |
|---|---|---|---|---|---|
| Morgan (r = 2, 2048 bits) | [0,0.2) | 4 | +2.7 | n/a | +6.2 |
| Morgan (r = 2, 2048 bits) | [0.2,0.4) | 43 | +4.6 | 5.4e-5 | +4.7 |
| Morgan (r = 2, 2048 bits) | [0.4,0.6) | 13 | +6.5 | 0.0015 | +7.4 |
| Morgan (r = 2, 2048 bits) | [0.6,1.0] | 27 | +2.2 | 0.022 | -0.3 |
| MACCS (167 bits) | [0.4,0.6) | 2 | +6.6 | n/a | +11.9 |
| MACCS (167 bits) | [0.6,1.0] | 85 | +4.0 | 8.2e-8 | +3.4 |
| RDKit topological | [0,0.2) | 4 | +7.5 | n/a | +7.1 |
| RDKit topological | [0.2,0.4) | 14 | +4.6 | 0.016 | +7.2 |
| RDKit topological | [0.4,0.6) | 19 | +4.7 | 0.012 | +2.8 |
| RDKit topological | [0.6,1.0] | 50 | +3.4 | 1.4e-4 | +2.6 |

| Fingerprint | Recovery measure | Spearman ρ | P |
|---|---|---|---|
| Morgan (r = 2, 2048 bits) | Best sample | -0.249 | 0.02 |
| Morgan (r = 2, 2048 bits) | t = 0.1 mean | -0.401 | 1.2e-4 |
| MACCS (167 bits) | Best sample | -0.098 | 0.37 |
| MACCS (167 bits) | t = 0.1 mean | -0.341 | 0.0012 |
| RDKit topological | Best sample | -0.232 | 0.031 |
| RDKit topological | t = 0.1 mean | -0.345 | 0.0011 |

Analyses use 87 zero-shot chains. Δ = PACo-Aptamer − RhoDesign (pp). Range P values are two-sided paired Wilcoxon tests, omitted for n < 10. Spearman correlations relate maximum Tanimoto similarity to recovery advantage.

**Supplementary Table 5. Leave one family out and family coverage results**

| Analysis | Training variant | Training chains | Zero-shot best (%) | Zero-shot t = 0.1 mean (%) | Δ best (pp) | P best | Δ t = 0.1 (pp) | P t = 0.1 |
|---|---|---|---|---|---|---|---|---|
| Leave one family out | Full training set | 768 | 73.3 | 63.3 | — | — | — | — |
| Leave one family out | − aminoglycosides | 640 | 71.5 | 60.7 | -1.9 | 2.7e-4 | -2.7 | 5.5e-4 |
| Leave one family out | − SAM/SAH | 713 | 71.9 | 62.2 | -1.5 | 0.0015 | -1.1 | 0.061 |
| Leave one family out | − NTPs | 748 | 72.3 | 62.4 | -1.0 | 0.097 | -0.9 | 0.11 |
| Leave one family out | − CLA | 744 | 72.9 | 63.6 | -0.5 | 0.5 | +0.3 | 0.65 |
| Leave one family out | − 2BA | 752 | 72.2 | 61.5 | -1.2 | 0.015 | -1.8 | 0.0028 |
| Leave one family out | Control retrained on all data | 768 | 73.4 | 64.5 | +0.0 | 0.91 | +1.2 | 0.07 |
| Family coverage | One family only | 128 | 71.2 | 62.7 | -2.1 | 1.4e-5 | -0.6 | 0.35 |
| Family coverage | Two families only | 183 | 73.5 | 64.1 | +0.2 | 0.69 | +0.8 | 0.25 |
| Family coverage | Three families only | 207 | 73.3 | 63.7 | -0.1 | 0.81 | +0.4 | 0.69 |
| Family coverage | Five families only | 243 | 73.7 | 62.7 | +0.4 | 0.36 | -0.6 | 0.25 |

Evaluations use 93 zero-shot chains. Δ = variant − full baseline (pp). P values are two-sided paired Wilcoxon tests.

**Supplementary Table 6. Ligand family definitions for the LOFO analysis**

| Family | Finetuning chains | CCD codes |
|---|---:|---|
| aminoglycosides | 128 | AM2+GET+NMY+PAR+RIO+TOB |
| SAM/SAH | 55 | SAH+SAM |
| NTP | 20 | ATP+CTP+GTP+UTP |
| CLA | 24 | CLA |
| 2BA | 16 | 2BA |
| GAI | 15 | GAI |
| flavins | 8 | FAD+FMN |

LOFO uses the five largest families. GAI and flavins were not included.

**Supplementary Table 7. Benchmarks using predicted backbones, by method**

The main analysis uses 193 chains, and the subset analysis uses 32 zero-shot chains.

| Method | Predicted best (%) | Predicted t = 0.1 (%) | Crystal best (%) | Crystal t = 0.1 (%) | Δ best (pp) | P best | Δ t = 0.1 (pp) | P t = 0.1 |
|---|---|---|---|---|---|---|---|---|
| PACo-Aptamer | 68.9 | 51.8 | 74.3 | 60.4 | -5.3 | 1.9e-18 | -8.6 | 1.1e-26 |
| RhoDesign | 61.4 | 48.0 | 68.3 | 54.9 | -6.9 | 5.3e-24 | -6.9 | 8.6e-17 |
| gRNAde | 50.7 | 35.0 | 52.2 | 36.4 | -1.6 | 1.8e-5 | -1.4 | 0.016 |
| NA-MPNN | 35.3 | 29.6 | 54.4 | 49.5 | -19.1 | 9.0e-28 | -19.8 | 1.0e-29 |
| AlignIF | 41.9 | 36.8 | 57.6 | 53.9 | -15.7 | 8.7e-20 | -17.1 | 1.8e-22 |

| Method | ρ(scTM, t = 0.1 recovery) | P |
|---|---|---|
| PACo-Aptamer | +0.216 | 0.0026 |
| RhoDesign | +0.057 | 0.44 |
| gRNAde | +0.233 | 0.0011 |
| NA-MPNN | +0.219 | 0.0022 |
| AlignIF | +0.671 | 1.4e-26 |

| Method | Predicted best (%) | Predicted t = 0.1 (%) | Crystal best (%) | Crystal t = 0.1 (%) |
|---|---|---|---|---|
| PACo-Aptamer | 74.6 | 56.1 | 80.4 | 67.1 |
| RhoDesign | 67.9 | 56.2 | 74.4 | 62.5 |

| Backbone | Δ best (pp) | P best | Δ t = 0.1 (pp) | P t = 0.1 |
|---|---|---|---|---|
| Predicted | +6.7 | 4.8e-6 | -0.1 | 0.42 |
| Crystal | +6.0 | 1.3e-5 | +4.6 | 0.0015 |

Method-level Δ = predicted − crystal, and zero-shot Δ = PACo-Aptamer − RhoDesign. P values are two-sided paired Wilcoxon tests on the same chains. gRNAde pools use corrected decoding. Correlations relate predicted-backbone scTM to t = 0.1 recovery on predicted backbones.

**Supplementary Table 8. Inference cost per method**

| Method | Wall time per chain (s), mean (range) | Time per sample (ms) | Throughput (samples/s) | Peak memory (MB), range |
|---|---|---|---|---|
| PACo-Aptamer | 2.77 (0.83–4.59) | 9.2 | 149.3 | 618–2081 |
| AlignIF | 9.49 (7.90–10.96) | 474.6 | 2.1 | 597–1241 |
| gRNAde | 6.40 (2.66–9.62) | 21.3 | 46.8 | 69–631 |
| NA-MPNN | 3.56 (3.38–3.85) | 355.9 | 2.8 | 560–802 |
| RhoDesign | 42.54 (17.25–97.73) | 154.7 | 6.5 | 154–313 |

Each method has six probes. PACo-Aptamer and AlignIF use crystal backbones, gRNAde and NA-MPNN use predicted backbones, and RhoDesign uses predicted backbones on an RTX 4090. PACo-Aptamer uses the two-model ensemble, and its single model is 1.96 times cheaper. Memory conventions differ across methods.

**Supplementary Table 9. Ablations of conditioning inputs**

The final training configuration and pools of 300 samples were used.

| Variant | Global best (%) | Global t = 0.1 (%) | Δ global best (pp) | P global best | Interface best (%) | Interface t = 0.1 (%) | Δ interface best (pp) | P interface best |
|---|---|---|---|---|---|---|---|---|
| Full model (ensemble of two members) | 74.6 | 59.7 | — | — | 85.6 | 60.0 | — | — |
| − ensemble (single model) | 71.4 | 59.7 | -3.2 | 1.3e-35 | 81.3 | 64.7 | -4.3 | 3.4e-5 |
| − secondary structure conditioning | 70.6 | 59.5 | -4.0 | 5.5e-40 | 81.1 | 58.5 | -4.5 | 5.8e-5 |
| − data augmentation | 70.7 | 59.9 | -3.9 | 3.7e-36 | 79.3 | 60.4 | -6.3 | 1.2e-5 |
| − ligand context | 70.8 | 59.1 | -3.8 | 7.3e-38 | 80.9 | 61.5 | -4.7 | 4.2e-5 |

Global and interface results use 477 and 115 chains, respectively. Δ = variant − full model (pp), and P values are two-sided paired Wilcoxon tests. Against the single model, interface best-sample changes without secondary structure, augmentation or ligand context are −0.2 (P = 0.57), −2.1 (P = 0.29) and −0.4 points (P = 0.40), respectively.

**Supplementary Table 10. Sequence overlap with the RNet training data by criterion and evaluation layer**

| Criterion | Test (n = 255) | Independent (n = 130) | Zero-shot (n = 37) | All (n = 422) |
|---|---|---|---|---|
| Exact match | 0 | 0 | 0 | 0 |
| High global similarity (≥ 0.8) | 8 | 0 | 3 | 11 |
| Embedded sequence (≤ 10% edits) | 178 | 27 | 22 | 227 |
| Flagged (deduplicated) | 178 (69.8%) | 27 (20.8%) | 22 (59.5%) | 227 (53.8%) |

All 11 highly similar records also meet the embedding criterion. Deduplication counts each record once per layer. Another 284 sequences share at least six 12-mers with RMDB but do not meet the overlap criteria and are classified as possible fragments.

**Supplementary Table 11. Evaluation of PACo-Aptamer training and ensemble variants**

Test results use 477 chains and 300-sample pools under the development convention, within 0.1 percentage points of final benchmark values. Validation uses 96 chains, with 20 samples at t = 0.1 for the mean or 20 per temperature for the best sample.

| Variant | Test best (%) | Test t = 0.1 mean (%) | Validation t = 0.1 mean | Validation best (60 samples) |
|---|---|---|---|---|
| Standard single model (seed 43) | 71.5 | 60.1 | 0.7230 | 0.7908 |
| Secondary structure dropout 0.15 | 71.5 | 59.4 | 0.7331 | 0.8061 |
| Base training replicate | 71.0 | 58.9 | n/a | n/a |
| Original base configuration | 70.4 | 58.5 | n/a | n/a |
| Alternative secondary structure dropout | 70.5 | 58.6 | n/a | n/a |
| Exponential moving average (0.999) | 69.9 | 57.8 | n/a | n/a |
| Seed 43 with dropout 0.15, trained after member selection | 71.4 | 58.0 | n/a | n/a |
| Extended stage 1 schedule | 70.4 | 57.6 | n/a | n/a |
| Stage 2 batch size 4 | 70.6 | 58.3 | n/a | n/a |
| Exploratory joint training | 69.6 | 57.0 | n/a | n/a |
| Reference configuration before the scan | 69.1 | 57.6 | n/a | n/a |
| Weight average of two checkpoints from the same stage 1 checkpoint | 70.9 | 58.9 | n/a | n/a |
| Weight average of three checkpoints | 69.6 | 55.4 | n/a | n/a |
| Weight average of four checkpoints from different stage 1 checkpoints | 66.2 | 51.4 | n/a | n/a |
| Output ensemble of two members, selected on validation | 74.6 | 59.6 | 0.7331 | 0.8263 |
| Output ensemble of three members, adding the base training replicate | 75.2 | 59.4 | 0.7256 | 0.8256 |
| Final output ensemble of two members without interface weighting | 74.6 | 59.7 | 0.7384 | 0.8281 |
| Final standard single model without interface weighting | n/a | n/a | 0.7343 | 0.7954 |
| Final single model with dropout 0.15 and without interface weighting | n/a | n/a | 0.7437 | 0.8135 |

The two-member ensemble averages probabilities from the standard and dropout-0.15 models. Selection used validation rather than test performance. Supplementary Note 7 gives training and checkpoint details.

## Supplementary References

References are numbered independently of the main text in order of first appearance in this supplement.

1. Wong, F. et al. Deep generative design of RNA aptamers using structural predictions. *Nat. Comput. Sci.* **4**, 829–839 (2024). https://doi.org/10.1038/s43588-024-00720-6

2. Joshi, C. K. et al. gRNAde: geometric deep learning for 3D RNA inverse design. In *Proc. 13th International Conference on Learning Representations* (2025).

3. Kubaney, A. et al. RNA sequence design and protein–DNA specificity prediction with NA-MPNN. Preprint at *bioRxiv* https://doi.org/10.1101/2025.10.03.679414 (2025).

4. Wang, S. et al. Structure-alignment-driven cross-graph modeling for functional RNA design. *Nat. Comput. Sci.* **6**, 826–841 (2026). https://doi.org/10.1038/s43588-026-01029-2

5. Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L. & Dror, R. Learning from protein structure with geometric vector perceptrons. In *Proc. 9th International Conference on Learning Representations* (2021).

6. Landrum, G. RDKit: open-source cheminformatics software. https://www.rdkit.org (2016).

7. Morgan, H. L. The generation of a unique machine description for chemical structures—a technique developed at Chemical Abstracts Service. *J. Chem. Doc.* **5**, 107–113 (1965). https://doi.org/10.1021/c160017a018

8. Dauparas, J. et al. Atomic context-conditioned protein sequence design using LigandMPNN. *Nat. Methods* **22**, 717–723 (2025). https://doi.org/10.1038/s41592-025-02626-1

9. Lorenz, R. et al. ViennaRNA Package 2.0. *Algorithms Mol. Biol.* **6**, 26 (2011). https://doi.org/10.1186/1748-7188-6-26

10. Chen, L. et al. AptaDB: a comprehensive database integrating aptamer–target interactions. *RNA* **30**, 189–199 (2024). https://doi.org/10.1261/rna.079784.123

11. Askari, A. et al. UTexas Aptamer Dataset. *Zenodo* https://doi.org/10.5281/zenodo.8387047 (2023).

12. Fu, L., Niu, B., Zhu, Z., Wu, S. & Li, W. CD-HIT: accelerated for clustering the next-generation sequencing data. *Bioinformatics* **28**, 3150–3152 (2012). https://doi.org/10.1093/bioinformatics/bts565

13. Shen, T. et al. Accurate RNA 3D structure prediction using a language model-based deep learning approach. *Nat. Methods* **21**, 2287–2298 (2024). https://doi.org/10.1038/s41592-024-02487-0

14. He, S. et al. Ribonanza: deep learning of RNA structure through dual crowdsourcing. Preprint at *bioRxiv* https://doi.org/10.1101/2024.02.24.581671 (2024).

15. Townley, J. et al. De novo design of RNA pseudoknots with deep learning. *Science* **393**, 931–937 (2026). https://doi.org/10.1126/science.aeg6829

16. Cordero, P., Lucks, J. B. & Das, R. An RNA Mapping DataBase for curating RNA structure mapping experiments. *Bioinformatics* **28**, 3006–3008 (2012). https://doi.org/10.1093/bioinformatics/bts554

17. Šošić, M. & Šikić, M. Edlib: a C/C++ library for fast, exact sequence alignment using edit distance. *Bioinformatics* **33**, 1394–1395 (2017). https://doi.org/10.1093/bioinformatics/btw753

18. Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024). https://doi.org/10.1038/s41586-024-07487-w

19. Ott, M. et al. fairseq: a fast, extensible toolkit for sequence modeling. In *Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)* 48–53 (2019). https://doi.org/10.18653/v1/N19-4009

20. Krogh, A. & Vedelsby, J. Neural network ensembles, cross validation, and active learning. In *Advances in Neural Information Processing Systems 7* 231–238 (1995).

21. Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems 30* 6402–6413 (2017).

22. Sennrich, R., Haddow, B. & Birch, A. Edinburgh neural machine translation systems for WMT 16. In *Proc. First Conference on Machine Translation* 371–376 (2016). https://doi.org/10.18653/v1/W16-2323

23. Frankle, J., Dziugaite, G. K., Roy, D. & Carbin, M. Linear mode connectivity and the lottery ticket hypothesis. In *Proc. 37th International Conference on Machine Learning* 3259–3269 (PMLR, 2020).

24. Durant, J. L., Leland, B. A., Henry, D. R. & Nourse, J. G. Reoptimization of MDL keys for use in drug discovery. *J. Chem. Inf. Comput. Sci.* **42**, 1273–1280 (2002). https://doi.org/10.1021/ci010132r

25. Bottaro, S. et al. Barnaba: software for analysis of nucleic acid structures and trajectories. *RNA* **25**, 219–231 (2019). https://doi.org/10.1261/rna.067678.118

26. Zhang, C., Shine, M., Pyle, A. M. & Zhang, Y. US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. *Nat. Methods* **19**, 1109–1115 (2022). https://doi.org/10.1038/s41592-022-01585-1
