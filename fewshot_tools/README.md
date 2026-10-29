# fewshot_tools

Helper modules imported by `notebooks/06_few_shot_transfer.ipynb` (Cell 1 stops if one is missing):

* `fewshot_core.py`: pair-disjoint support/query split, support-only model selection, adapters, metrics, paired bootstrap
* `export_features.py`: exports the frozen source-model representations
* `source_runtime.py`: data preparation and model definition shared with notebook 04
* `test_fewshot.py`: self-test on synthetic data (a software check, not a biomedical result)

Copy this folder to `DualGATSynergy/fewshot_tools/` on Drive. Notebook 07 is self-contained and writes its own helpers.
