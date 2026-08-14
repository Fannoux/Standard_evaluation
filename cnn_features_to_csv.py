#!/usr/bin/env python3
"""
Turn a CNN evaluation output dir into a labelled feature table that standard_eval.py can read
(framing A: the CNN's penultimate features scored by the SAME protocol as the other methods).

It merges:
  <cnn_eval_dir>/encodings_evaluation.csv   (penultimate features, indexed by image path)
  <cnn_eval_dir>/results_evaluation.csv      (has img_path + label_categorical)
into one CSV with a 'label' column.

Produce the CNN eval dir FIRST, on the VALIDATION split (keeps the test set untouched):
  python evaluation.py -Y params_file/param_finetune.yml -M <best_model.pth> -S validation

Then:
  python cnn_features_to_csv.py --cnn-eval-dir <that_dir> --out results/cnn_val_features.csv
  python standard_eval.py --csv results/cnn_val_features.csv --label-col label --name CNN_features_val
"""
import argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cnn-eval-dir', required=True, help='dir containing encodings_evaluation.csv + results_evaluation.csv')
    ap.add_argument('--out', default='results/cnn_features_labelled.csv')
    args = ap.parse_args()

    enc = pd.read_csv(os.path.join(args.cnn_eval_dir, 'encodings_evaluation.csv'), index_col=0)
    res = pd.read_csv(os.path.join(args.cnn_eval_dir, 'results_evaluation.csv'))
    path_col = 'img_path' if 'img_path' in res.columns else res.columns[0]
    labels = res.set_index(path_col)['label_categorical']

    df = enc.copy()
    df['label'] = labels.reindex(df.index).values
    n_before = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.reset_index(drop=True).to_csv(args.out, index=False)
    print(f"[OK] {len(df)}/{n_before} samples, {df.shape[1]-1} features -> {args.out}")
    print(f"  next: python standard_eval.py --csv {args.out} --label-col label --name CNN_features_<set>")


if __name__ == '__main__':
    main()
