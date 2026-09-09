#!/usr/bin/env python3
"""
Standardized cross-method evaluation on any representation (features or learned latents).
Trains LogisticRegression + RandomForest, reports multiclass and binary tasks with macro-F1,
weighted-F1 and (quadratic-weighted) Cohen's kappa.

    python standard_eval.py --train-csv train.csv --eval-csv val.csv --name CNN_val   # hold-out
    python standard_eval.py --csv features.csv --label-col label --name RegionProps   # 5-fold CV

Feature CSVs use the shared format (data_id, f0..fN, label) from features_to_csv.py.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
import os


def make_clfs():
    """Fresh LogReg + RandomForest instances."""
    return [('LogReg', LogisticRegression(max_iter=3000)),
            ('RF', RandomForestClassifier(n_estimators=300, random_state=0))]


def res_row(name, task, nm, protocol, y_true, y_pred, dims, n_train, n_eval):
    binary = (task == 'binary')
    r = dict(method=name, task=task, classifier=nm, protocol=protocol,
             n_train=n_train, n_eval=n_eval, dims=dims,
             acc=accuracy_score(y_true, y_pred),
             macroF1=f1_score(y_true, y_pred, average='macro'),
             weightedF1=f1_score(y_true, y_pred, average='weighted'),
             quadKappa=cohen_kappa_score(y_true, y_pred, weights=None if binary else 'quadratic'))
    return r


def output_print(nm, r, binary, y_true=None, y_pred=None):
    if binary:
        extra = f"F1(affected)={f1_score(y_true, y_pred):.3f} " if y_true is not None else ""
        print(f"    {nm:7s} acc={r['acc']:.3f} {extra}macroF1={r['macroF1']:.3f} kappa={r['quadKappa']:.3f}")
    else:
        print(f"    {nm:7s} acc={r['acc']:.3f} macroF1={r['macroF1']:.3f} "
              f"weightedF1={r['weightedF1']:.3f} quadKappa={r['quadKappa']:.3f}")


def report(X, y, name):
    """5-fold stratified CV within a single set; returns (rows, confusion matrices)."""
    X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    n, dims = len(y), X.shape[1]
    rows, cms = [], {}
    print(f"\n=== {name}  [CV5]  (n={n}, dims={dims}, classes={np.bincount(y).tolist()}) ===")
    print("  5-class:")
    for nm, clf in make_clfs():
        yp = cross_val_predict(clf, X, y, cv=cv)
        r = res_row(name, '5class', nm, 'cv5', y, yp, dims, n, n)
        rows.append(r); cms[f'5class_{nm}'] = confusion_matrix(y, yp)
        output_print(nm, r, binary=False)
    yb = (y > 0).astype(int)
    print("  binary (class 0 vs rest):")
    for nm, clf in make_clfs():
        yp = cross_val_predict(clf, X, yb, cv=cv)
        r = res_row(name, 'binary', nm, 'cv5', yb, yp, dims, n, n)
        rows.append(r); cms[f'binary_{nm}'] = confusion_matrix(yb, yp)
        output_print(nm, r, binary=True, y_true=yb, y_pred=yp)
    return rows, cms


def report_holdout(Xtr, ytr, Xte, yte, name):
    """Fit on train, evaluate on held-out set; scaler fit on train only."""
    sc = StandardScaler().fit(np.asarray(Xtr, dtype=float))
    Xtr = sc.transform(np.asarray(Xtr, dtype=float))
    Xte = sc.transform(np.asarray(Xte, dtype=float))
    ytr = np.asarray(ytr).astype(int); yte = np.asarray(yte).astype(int)
    ntr, nte, dims = len(ytr), len(yte), Xtr.shape[1]
    rows, cms = [], {}
    print(f"\n=== {name}  [HOLD-OUT]  (train n={ntr}, eval n={nte}, dims={dims}, "
          f"train classes={np.bincount(ytr).tolist()}, eval classes={np.bincount(yte).tolist()}) ===")
    print("  5-class:")
    for nm, clf in make_clfs():
        clf.fit(Xtr, ytr); yp = clf.predict(Xte)
        r = res_row(name, '5class', nm, 'holdout', yte, yp, dims, ntr, nte)
        rows.append(r); cms[f'5class_{nm}'] = confusion_matrix(yte, yp)
        output_print(nm, r, binary=False)
    ytr_b, yte_b = (ytr > 0).astype(int), (yte > 0).astype(int)
    print("  binary (class 0 vs rest):")
    for nm, clf in make_clfs():
        clf.fit(Xtr, ytr_b); yp = clf.predict(Xte)
        r = res_row(name, 'binary', nm, 'holdout', yte_b, yp, dims, ntr, nte)
        rows.append(r); cms[f'binary_{nm}'] = confusion_matrix(yte_b, yp)
        output_print(nm, r, binary=True, y_true=yte_b, y_pred=yp)
    return rows, cms


def load_csv(path, label_col, drop_cols, id_col='data_id'):
    """Load a feature CSV; returns (X, y, data_ids, feature_columns)."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise SystemExit(f"[ERR] label column '{label_col}' not in {path} (cols: {list(df.columns)[:8]}...)")
    y = df[label_col].values
    if id_col in df.columns:
        ids = df[id_col].astype(str).values
    else:
        print(f"[WARN] id column '{id_col}' not in {path}; falling back to positional row indices, "
              f"so the train/eval leakage check is unreliable. Pass --id-col for the real id column.")
        ids = pd.Series(np.arange(len(df))).astype(str).values
    drop = set(drop_cols) | {label_col, id_col}
    X = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes('number').fillna(0)
    return X.values, y, ids, list(X.columns)


def save(rows, cms, name, out):
    os.makedirs(out, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(os.path.join(out, f'metrics_{safe}.csv'), index=False)
    for k, cm in cms.items():
        np.savetxt(os.path.join(out, f'confusion_{safe}_{k}.csv'), cm, fmt='%d', delimiter=',')
    master = os.path.join(out, 'all_methods_comparison.csv')
    prev = pd.read_csv(master) if os.path.exists(master) else pd.DataFrame()
    if len(prev):                                   # re-running a name overwrites its rows
        key = ['method', 'task', 'classifier']
        prev = prev[~prev[key].apply(tuple, 1).isin(run_df[key].apply(tuple, 1))]
    pd.concat([prev, run_df], ignore_index=True).to_csv(master, index=False)
    print(f"\n[OK] saved metrics_{safe}.csv (+ confusion matrices) -> {out}/")
    print(f"[OK] master comparison table -> {master}")


def main():
    ap = argparse.ArgumentParser()
    # hold-out
    ap.add_argument('--train-csv', help='features CSV to fit the classifier on (train set)')
    ap.add_argument('--eval-csv', help='features CSV to evaluate on (held-out set)')
    # CV
    ap.add_argument('--csv'); ap.add_argument('--latents'); ap.add_argument('--labels')
    ap.add_argument('--label-col', default='label')
    ap.add_argument('--id-col', default='data_id',
                    help='data-id column (default: data_id); used for the leakage check and '
                         'dropped from the features')
    ap.add_argument('--drop-cols', nargs='*', default=['mask', 'stem', 'CO6', 'set'])
    ap.add_argument('--name', default='representation')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                    help='dir for saved metrics (default: ./results next to this script)')
    args = ap.parse_args()

    if args.train_csv and args.eval_csv:                             # hold-out
        Xtr, ytr, idtr, cols_tr = load_csv(args.train_csv, args.label_col, args.drop_cols, args.id_col)
        Xte, yte, idte, cols_te = load_csv(args.eval_csv, args.label_col, args.drop_cols, args.id_col)
        if cols_tr != cols_te:
            raise SystemExit(f"[ERR] train/eval feature columns differ ({len(cols_tr)} vs {len(cols_te)}) "
                             "-- both must come from the same output convention.")
        overlap = set(idtr) & set(idte)
        if overlap:
            print(f"[WARN] LEAKAGE: {len(overlap)} samples appear in both train and eval "
                  f"(first few: {sorted(overlap)[:3]}). They must be disjoint.")
        rows, cms = report_holdout(Xtr, ytr, Xte, yte, args.name)

    elif args.latents and args.labels:                              # CV (npy)
        print("[INFO] CV-on-one-set protocol")
        rows, cms = report(np.load(args.latents), np.load(args.labels), args.name)

    elif args.csv:                                                  # CV (csv)
        print("[INFO] CV-on-one-set protocol")
        X, y, _, _ = load_csv(args.csv, args.label_col, args.drop_cols, args.id_col)
        rows, cms = report(X, y, args.name)

    else:
        ap.error('give --train-csv AND --eval-csv (hold-out), or --csv / --latents+--labels (CV)')

    save(rows, cms, args.name, args.out)


if __name__ == '__main__':
    main()
