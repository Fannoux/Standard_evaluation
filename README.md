# Standard evaluation

Downstream evaluation of image representations (RegionProps, ShapeEmbed, VAE, CNN) for larval
spinal-kink severity, on a common protocol so the methods are directly comparable.

## Scripts

- `standard_eval.py` — classification protocol: standardize features, fit LogisticRegression and
  RandomForest, and score the 5-class (SC0–4) and binary (SC0 vs any-kink) tasks with macro-F1,
  weighted-F1 and quadratic-weighted Cohen's kappa. Supports a train/hold-out split
  (`--train-csv/--eval-csv`) or cross-validation (`--csv/--latents`).
- `features_to_csv.py` — adapter from each method's output to the shared `data_id, f0..fN, label`
  table.
- `compare_representations.py` — representation-level comparison (CKA, cross-prediction R²).
- `continuous_vs_kinks.py` — correlation between the expected severity `E[k]` and the raw kink count.
- `heritability_cka.py` — alignment between each representation and the genetic relationship matrix.
- `occlusion/`, `gradcam/` — interpretability analyses (see their own READMEs).
- `RegionProps/` — hand-crafted shape-descriptor baseline.

## Usage

```bash
# latents (ShapeEmbed / VAE)
python standard_eval.py --latents <X.npy> --labels <y.npy> --name <NAME>
# feature table (RegionProps / CNN)
python standard_eval.py --csv <feats.csv> --label-col label --name <NAME>
# hold-out (fit on train, evaluate on held-out)
python standard_eval.py --train-csv <train.csv> --eval-csv <eval.csv> --name <NAME>
```

Feature tables share the format `data_id, f0..fN, label`.
