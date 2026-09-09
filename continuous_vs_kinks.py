#!/usr/bin/env python3
"""
Continuous-label vs raw-count analysis: does the model track the underlying continuum even when
discrete multiclass accuracy is low? Fits LogReg + RandomForest on train, forms the expected value
E[k] = sum_c class_c * P(class_c) per test sample, and correlates it (Spearman) with the raw
kinks_number from the manifest (joined by data_id).

    python continuous_vs_kinks.py --train-csv train.csv --eval-csv test.csv \
        --manifest manifest.csv --name ShapeEmbed --out results/continuous

Feature CSVs use the shared format (data_id, f0..fN, label). Manifest needs image_path + kinks_number.
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
    ids = df['data_id'].astype(str).values if 'data_id' in df else np.arange(len(df)).astype(str)
    X = df.drop(columns=[c for c in ('data_id', 'label') if c in df.columns]).select_dtypes('number').fillna(0).values
    return X, y, ids


def expected_severity(clf, Xte):
    P = clf.predict_proba(Xte)                      # (n, n_classes_seen)
    ks = clf.classes_.astype(float)                 # class labels seen in train
    return P @ ks                                   # E[k] per sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--eval-csv', required=True)
    ap.add_argument('--manifest', required=True, help='split manifest (image_path + kinks_number)')
    ap.add_argument('--kink-col', default='kinks_number')
    ap.add_argument('--name', default='method')
    ap.add_argument('--out', default='results/continuous')
    a = ap.parse_args()

    Xtr, ytr, _ = load(a.train_csv)
    Xte, yte, idte = load(a.eval_csv)
    sc = StandardScaler().fit(Xtr)                  # fit on train only
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    # raw kink count per test sample, joined by data_id == basename(image_path)
    man = pd.read_csv(a.manifest)
    man['stem'] = man['image_path'].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    lut = dict(zip(man['stem'], man[a.kink_col]))
    kink = np.array([lut.get(str(i), np.nan) for i in idte], dtype=float)
    matched = ~np.isnan(kink)
    print(f"[INFO] {matched.sum()}/{len(idte)} test samples matched to manifest kink count")

    rows = []
    out = pd.DataFrame({'data_id': idte, 'SC_label': yte, 'kink_count': kink})
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

    # scatter: E[k] (RF) vs raw kink count, coloured by class label
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d = out[matched]
        fig, ax = plt.subplots(figsize=(5, 4))
        sctr = ax.scatter(d['kink_count'], d['E_k_RF'], c=d['SC_label'], cmap='viridis', s=10, alpha=0.5)
        ax.set_xlabel('raw kink count'); ax.set_ylabel('expected value E[k] (RF)')
        ax.set_title(f'{a.name}: continuous prediction vs kink count')
        fig.colorbar(sctr, label='class label')
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f'scatter_{a.name}.png'), dpi=300)
        print(f"[OK] -> {a.out}/scatter_{a.name}.png")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")


if __name__ == '__main__':
    main()
