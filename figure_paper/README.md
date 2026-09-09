# figure_paper

Scripts that generate the paper's tables and figures. Kept **as-is** — they carry the study's
specific captions, class labels, and result-directory layout, so they are project-specific rather
than general-purpose.

- `make_results_table.py`  — LaTeX held-out test-set results table
- `make_ood_table.py`      — LaTeX validation-vs-test (OOD) table
- `plot_holdout_results.py` — held-out result plots

Each reads from a `HOLDOUT_TestSet_Results/` (and `TrainVal_Results/`) directory **next to the
script**; edit the CONFIG block at the top of each to point at your results.
