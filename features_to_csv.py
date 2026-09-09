#!/usr/bin/env python3
"""
Adapter: turn any method's saved representation into one standard table (data_id, f0..fN, label)
that standard_eval.py can score and that joins to the shared split by data_id.

    python features_to_csv.py --cnn-eval-dir <dir> --out cnn_features.csv                    # CNN eval dir
    python features_to_csv.py --latents mu.npy --labels y.npy --ids ids.npy --out vae.csv    # npy latents
    python features_to_csv.py --in-csv feats.csv --id-col stem --label-col label --out rp.csv  # raw CSV

--id-mode stem (default) reduces each identifier to basename-without-extension so all methods
share the same data_id key. Feature columns are renamed f0..fN; rows with a missing label are dropped.
"""
import argparse, os
import numpy as np
import pandas as pd


def _to_data_id(series, id_mode):
    s = series.astype(str)
    if id_mode == 'stem':
        s = s.map(lambda x: os.path.splitext(os.path.basename(x))[0])
    return s


def _load_ids(path, n):
    if path is None:
        print("[WARN] no --ids given: falling back to positional index; "
              "rows will not join to other methods by data_id.")
        return pd.Series([f'row{i}' for i in range(n)])
    if path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True).ravel()
    else:  # one id per line
        with open(path) as fh:
            arr = [ln.strip() for ln in fh if ln.strip()]
    if len(arr) != n:
        raise SystemExit(f"[ERR] --ids has {len(arr)} rows but features have {n}")
    return pd.Series(arr)


def _assemble(feat_df, data_id, label, out, id_mode):
    """Write the standard table from numeric features and aligned data_id/label Series."""
    feat_df = feat_df.reset_index(drop=True)
    feat_df.columns = [f'f{i}' for i in range(feat_df.shape[1])]
    df = feat_df.copy()
    df.insert(0, 'data_id', _to_data_id(pd.Series(data_id).reset_index(drop=True), id_mode))
    df['label'] = pd.Series(label).reset_index(drop=True).values
    n_before = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    if df['data_id'].duplicated().any():
        print(f"[WARN] {int(df['data_id'].duplicated().sum())} duplicate data_id values")
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)}/{n_before} samples, {df.shape[1]-2} features -> {out}")
    print(f"  next: python standard_eval.py --csv {out} --label-col label --name <NAME>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cnn-eval-dir', help='dir with encodings_evaluation.csv + results_evaluation.csv')
    ap.add_argument('--latents'); ap.add_argument('--labels'); ap.add_argument('--ids')
    ap.add_argument('--vae-pkl', help='VAE latents pkl (list of per-batch dicts)')
    ap.add_argument('--pkl-label-col', default='severity_score_adjusted')
    ap.add_argument('--in-csv'); ap.add_argument('--id-col'); ap.add_argument('--label-col')
    ap.add_argument('--drop-cols', nargs='*', default=['mask', 'CO6', 'set'])
    ap.add_argument('--id-mode', choices=['stem', 'raw'], default='stem')
    ap.add_argument('--out', default='results/features_labelled.csv')
    args = ap.parse_args()

    if args.cnn_eval_dir:                                   # mode 1: CNN eval dir
        enc = pd.read_csv(os.path.join(args.cnn_eval_dir, 'encodings_evaluation.csv'), index_col=0)
        res = pd.read_csv(os.path.join(args.cnn_eval_dir, 'results_evaluation.csv'))
        path_col = 'img_path' if 'img_path' in res.columns else res.columns[0]
        labels = res.set_index(path_col)['label_categorical'].reindex(enc.index)
        _assemble(enc, enc.index.to_series(), labels.values, args.out, args.id_mode)

    elif args.vae_pkl:                                     # mode 2b: the VAE pkl
        import pickle
        with open(args.vae_pkl, 'rb') as fh:
            data = pickle.load(fh)                         # list of per-batch dicts
        mus, ids, labs = [], [], []
        for b in data:
            mus.append(np.asarray(b['mu']))
            ids += [str(p) for p in b['img_paths']]
            labs += list(np.asarray(b[args.pkl_label_col]).ravel())
        _assemble(pd.DataFrame(np.concatenate(mus, axis=0)), pd.Series(ids), labs, args.out, args.id_mode)

    elif args.latents and args.labels:                     # mode 2: npy latents
        X = np.load(args.latents); y = np.load(args.labels, allow_pickle=True).ravel()
        ids = _load_ids(args.ids, len(X))
        _assemble(pd.DataFrame(X), ids, y, args.out, args.id_mode)

    elif args.in_csv and args.id_col and args.label_col:   # mode 3: raw csv
        df = pd.read_csv(args.in_csv)
        data_id = df[args.id_col]; label = df[args.label_col]
        drop = set(args.drop_cols) | {args.id_col, args.label_col}
        feats = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes('number').fillna(0)
        _assemble(feats, data_id, label, args.out, args.id_mode)

    else:
        ap.error('give one of: --cnn-eval-dir | --latents/--labels | --in-csv/--id-col/--label-col')


if __name__ == '__main__':
    main()
