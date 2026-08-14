#!/usr/bin/env python3
"""
Continuous-severity vs raw-kink-count analysis (does the model track the underlying continuum
even when discrete 5-class accuracy is low?).

For one method: fit LogReg + RandomForest on TRAIN features, predict_proba on TEST, form the
expected severity  E[k] = sum_c class_c * P(class_c)  per test fish, and correlate it (Spearman)
with the RAW kink count (from the manifest, joined by fish_id) -- NOT the binned SC label.
A decent rho with low accuracy = the binning discards signal the model actually captures.

Usage:
  python continuous_vs_kinks.py --train-csv features_train.csv --eval-csv features_test.csv \
      --manifest <canonical_manifest.csv> --name ShapeEmbed --out results/continuous

Feature CSVs are the shared format (fish_id, f0..fN, label). Manifest must have image_path +
kinks_number (fish_id joins on basename(image_path) without extension).
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load(path):
    df = pd.read_csv(path)
    y = df['label'].astype(int).values
    ids = df['fish_id'].astype(str).values if 'fish_id' in df else np.arange(len(df)).astype(str)
    X = df.drop(columns=[c for c in ('fish_id', 'label') if c in df.columns]).select_dtypes('number').fillna(0).values
    return X, y, ids


def expected_severity(clf, Xte):
    P = clf.predict_proba(Xte)                      # (n, n_classes_seen)
    ks = clf.classes_.astype(float)                 # actual class labels present in train
    return P @ ks                                   # E[k] per fish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--eval-csv', required=True)
    ap.add_argument('--manifest', required=True, help='canonical split manifest (image_path + kinks_number)')
    ap.add_argument('--kink-col', default='kinks_number')
    ap.add_argument('--name', default='method')
    ap.add_argument('--out', default='results/continuous')
    a = ap.parse_args()

    Xtr, ytr, _ = load(a.train_csv)
    Xte, yte, idte = load(a.eval_csv)
    sc = StandardScaler().fit(Xtr)                  # fit on train only
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    # raw kink count per test fish, joined by fish_id == basename(image_path)
    man = pd.read_csv(a.manifest)
    man['stem'] = man['image_path'].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    lut = dict(zip(man['stem'], man[a.kink_col]))
    kink = np.array([lut.get(str(i), np.nan) for i in idte], dtype=float)
    matched = ~np.isnan(kink)
    print(f"[INFO] {matched.sum()}/{len(idte)} test fish matched to manifest kink count")

    rows = []
    out = pd.DataFrame({'fish_id': idte, 'SC_label': yte, 'kink_count': kink})
    for nm, clf in [('LogReg', LogisticRegression(max_iter=1000)),
                    ('RF', RandomForestClassifier(random_state=42))]:
        clf.fit(Xtr, ytr)
        Ek = expected_severity(clf, Xte)
        out[f'E_k_{nm}'] = Ek
        d = out[matched]
        rho_kink = d[f'E_k_{nm}'].corr(d['kink_count'], method='spearman')
        rho_lbl = d[f'E_k_{nm}'].corr(d['SC_label'], method='spearman')
        rows.append((nm, rho_kink, rho_lbl))
        print(f"[{a.name}/{nm}]  Spearman rho(E[k], kink_count) = {rho_kink:.3f}   "
              f"rho(E[k], SC_label) = {rho_lbl:.3f}")

    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, f'continuous_{a.name}.csv'), index=False)
    pd.DataFrame(rows, columns=['classifier', 'rho_Ek_vs_kink', 'rho_Ek_vs_SClabel']).to_csv(
        os.path.join(a.out, f'spearman_{a.name}.csv'), index=False)

    # scatter: E[k] (RF) vs raw kink count, coloured by SC label
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d = out[matched]
        fig, ax = plt.subplots(figsize=(5, 4))
        sctr = ax.scatter(d['kink_count'], d['E_k_RF'], c=d['SC_label'], cmap='viridis', s=10, alpha=0.5)
        ax.set_xlabel('raw kink count'); ax.set_ylabel('expected severity E[k] (RF)')
        ax.set_title(f'{a.name}: continuous prediction vs kink count')
        fig.colorbar(sctr, label='SC label')
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f'scatter_{a.name}.png'), dpi=300)
        print(f"[OK] -> {a.out}/scatter_{a.name}.png")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")


if __name__ == '__main__':
    main()
