# DualGATSynergy

Code and notebooks accompanying the manuscript on **DualGATSynergy**, a weight-shared graph-attention model with inter-molecular cross-attention (IMCA) for drug-synergy prediction under a cold-drug protocol (DrugComb v1.5 + O'Neil; 83,848 records, 309 compound identifiers).

> **Archive / citation:** https://doi.org/10.5281/zenodo.`<TODO: Zenodo concept DOI>`
> **Status:** the evaluation notebooks (`evaluate_saved` mode) load the trained models and reproduce the reported numbers; retraining is optional.

## Repository layout

The notebooks are written for Google Colab and expect a project folder `ROOT = /content/drive/MyDrive/DualGATSynergy` (set in Cell 2 of each notebook). Copy this repository's content into that Drive folder (the `notebooks/` folder can be anywhere).

```
notebooks/            the eight notebooks listed below
fewshot_tools/        helper modules used by notebook 06 (copy to the project folder on Drive)
environment_colab_freeze.txt   full pip freeze of the Colab environment
data/                 cell_lines.csv, drugs.csv, smiles_cache.csv, oneil_summary_idx.csv, drug_to_chemberta.pt
                      drugcomb_full_v1.5.csv  -> NOT in the repository, download it (see "Data")
splits/               cold_drug_5fold.json, structure_grouped_5fold.json, structure_groups.json,
                      duplicate_compounds.json, cell_line_to_idx.json
results/              shipped summary tables (reported values the notebooks compare against)
predictions/          shipped test-set predictions
figures/              figures produced by the notebooks
checkpoints/ checkpoints_retrained/ cross_dataset/    trained models -> Zenodo (see "Checkpoints")
```

## Notebooks (in the order of the paper)

| Notebook | What it does | Manuscript |
|---|---|---|
| `01_main_cold_drug` | Data construction, graph/ChemBERTa features, five-fold cold-drug training/evaluation, RF/XGBoost baselines (with and without cell-line input), structure-level leakage audit, interpretability | Sections 5.1-5.3, 5.8; Tables 2, 4, 5; Figures 4-6 |
| `02_ablation_study` | Four architecture variants (no weight sharing, no IMCA, unidirectional IMCA, structure-only) x five folds | Section 5.4, Table 9 |
| `02b_context_stream_ablation` | Three further variants (no ChemBERTa, no cell line, graph-only concatenation) under the same protocol; completes the eight rows of Table 9 | Section 5.4, Table 9 |
| `03_order_sensitivity` | Evaluates the saved models with drug order (A,B) and (B,A); no training | Section 5.5, Table 10, Figure 3 |
| `04_dataset_composition_cross_dataset` | Dataset documentation, label standardization, per-cohort results, DrugComb <-> O'Neil cross-dataset validation | Sections 5.1, 5.6, 5.7; Tables 1, 11, 12; Figure 2 |
| `05_structure_grouped_cv` | Structure-grouped five-fold cross-validation (289 groups), retrained, with RF/XGBoost | Structure-grouped results (Tables 3, 6-8) |
| `06_few_shot_transfer` | Few-label target adaptation (16/32/64/128 labels, pair-disjoint query): source-only, residual Ridge, target-only fingerprint Ridge | Section 5.7, Table 13 |
| `07_fewshot_head_finetuning` | Supervised fine-tuning of the prediction head with the same support/query sets | Section 5.7, Table 13 (head fine-tuning row) |

Run order: `01` first (it writes the models and results the others read), then `02`, `02b`, `03`, `04`, `05` in any order; `07` needs the output of `06` (`fewshot_extension_v1/`).

Most notebooks have a `MODE` switch: `show_reported` (prints the shipped table, seconds), `evaluate_saved` (default; loads checkpoints, CPU is enough), `train_and_evaluate` (retrains; resumable).

## Data

* **DrugComb v1.5** (https://drugcomb.org): download the full table and save it as `data/drugcomb_full_v1.5.csv`.
* **O'Neil screen:** the processed summary used here is `data/oneil_summary_idx.csv`.
* Compound identifiers, SMILES mapping and cell-line indices are in `data/`; the fold definitions are in `splits/`.

## Checkpoints

Trained checkpoints are archived in the Zenodo record above. Download them and unpack into `checkpoints/`, `checkpoints_retrained/` and `cross_dataset/` inside the project folder.

## Environment

Colab (T4 GPU for training; CPU is enough for `evaluate_saved`). The first cells of each notebook install the packages; the versions used are listed in `requirements.txt`, and the complete Colab `pip freeze` is archived in `environment_colab_freeze.txt`. This freeze was written when the few-label notebook (06) was run; RDKit, which the structure-level audit depends on, is pinned to the same version in notebook 01.

## Notes on interpretation

* Few-label experiments (06, 07) are **exploratory**. Standard deviations mix source seeds and support draws and are not independent runs; bootstrap intervals resample query pairs and ignore dependence between pairs that share a drug.
* Notebooks 06 and 07 do not use any query label for fitting or model selection.

## License and citation

MIT License (see `LICENSE`). Please cite via `CITATION.cff` or the Zenodo record.
