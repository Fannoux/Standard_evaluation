#!/usr/bin/env python3
"""
Representation-level comparison of two methods on the same samples (no classifier).
Reports linear CKA, cross-prediction R^2 (A predicts each feature of B), and top canonical
correlation (CCA). Inputs: two feature CSVs in the shared format (data_id, features..., [label]),
aligned by data_id. Convention: A = source, B = target; R^2 is A->B (--both-directions adds B->A).
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA

ID_CANDIDATES = ['data_id', 'stem', 'id']
NON_FEATURE = {'mask', 'CO6', 'set', 'generation', 'label', 'label_categorical',
               'label_regression', 'severity_score', 'severity_score_adjusted'}


def _stem(s):
    return os.path.splitext(os.path.basename(str(s)))[0]


def load_features(path, id_col=None, label_col='label'):
    """Read a feature CSV; returns (features indexed by data_id, label Series or None)."""
    df = pd.read_csv(path)
    idc = id_col or next((c for c in ID_CANDIDATES if c in df.columns), None)
    if idc is None:
        raise SystemExit(f"[ERR] no id column in {path} (looked for {ID_CANDIDATES}; pass --id-col)")
    data_id = df[idc].map(_stem)
    label = df[label_col] if label_col in df.columns else None
    drop = NON_FEATURE | {idc, label_col} | set(ID_CANDIDATES)
    feats = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes('number').fillna(0)
    feats.index = data_id.values
    lab = pd.Series(label.values, index=data_id.values) if label is not None else None
    return feats, lab


def lin_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((np.linalg.norm(Y.T @ X) ** 2) / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def cross_pred_r2(Xa, B_df, alpha):
    """R^2 of predicting each column of B from A with 5-fold ridge; returns {feature: r2}."""
    r2 = {}
    B = B_df.values.astype(float)
    for j, f in enumerate(B_df.columns):
        pred = cross_val_predict(Ridge(alpha=alpha), Xa, B[:, j], cv=5)
        r2[str(f)] = float(r2_score(B[:, j], pred))
    return r2


def barchart(r2, src, dst, out, top_n):
    d = pd.DataFrame(sorted(r2.items(), key=lambda kv: kv[1]), columns=['feature', 'cv_r2'])
    if len(d) > top_n:
        d = d.tail(top_n)
    colors = ['#007786' if v > 0.8 else '#9ca3af' for v in d['cv_r2']]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d) + 1)))
    ax.barh(d['feature'], d['cv_r2'].clip(lower=0), color=colors)
    ax.set_xlabel(f'CV $R^2$  ({src} latent $\\rightarrow$ {dst} feature)')
    ax.axvline(0.8, ls='--', c='grey', lw=1); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def corr_clustermap(A_df, B_df, name_a, name_b, out):
    """Clustered heatmap of feature-wise Pearson r between A and B; saves .png + .csv."""
    import seaborn as sns
    A = StandardScaler().fit_transform(A_df.values.astype(float))
    B = StandardScaler().fit_transform(B_df.values.astype(float))
    C = np.clip((B.T @ A) / len(A), -1, 1)                  # standardized dot / n = Pearson r
    dfC = pd.DataFrame(C, index=B_df.columns, columns=A_df.columns).fillna(0).drop('Unnamed: 0', axis=0, errors='ignore')
    dfC.to_csv(f'{out}/feature_correlation_{name_a}_vs_{name_b}.csv')
    nb, na = dfC.shape
    g = sns.clustermap(
        dfC, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        figsize=(min(0.45 * na + 4, 22), min(0.4 * nb + 4, 18)),
        row_cluster=(nb >= 2), col_cluster=(na >= 2),       # need >=2 points to cluster an axis
        annot=(nb * na <= 400), fmt='.2f', annot_kws={'size': 6},
        xticklabels=(na <= 60), yticklabels=True,
        dendrogram_ratio=(0.12, 0.12), cbar_pos=(0.02, 0.83, 0.03, 0.14),
        cbar_kws={'label': 'Pearson r'},
    )
    g.ax_heatmap.set_xlabel(f'{name_a} features'); g.ax_heatmap.set_ylabel(f'{name_b} features')
    g.ax_heatmap.tick_params(axis='x', labelsize=7); g.ax_heatmap.tick_params(axis='y', labelsize=8)
    g.figure.suptitle(f'{name_a} vs {name_b}: feature-wise correlation (clustered;  $R^2 = r^2$)')
    g.savefig(f'{out}/feature_correlation_clustermap_{name_a}_vs_{name_b}.png', dpi=300)
    plt.close(g.figure)


def analyse(name_a, name_b, A, B, out, alpha, top_n, both, note, labels):
    Xa = StandardScaler().fit_transform(A.values.astype(float))
    Xb = StandardScaler().fit_transform(B.values.astype(float))
    Xb_df = pd.DataFrame(Xb, columns=B.columns)

    cka = lin_cka(Xa, Xb)
    r2_ab = cross_pred_r2(Xa, Xb_df, alpha)
    ncomp = min(2, Xa.shape[1], Xb.shape[1])
    cca = CCA(n_components=ncomp).fit(Xa, Xb)
    Ux, Vx = cca.transform(Xa, Xb)
    canon = [float(np.corrcoef(Ux[:, k], Vx[:, k])[0, 1]) for k in range(ncomp)]

    pd.DataFrame(sorted(r2_ab.items(), key=lambda kv: -kv[1]), columns=['feature', 'cv_r2']) \
        .to_csv(f'{out}/cross_prediction_r2_{name_a}_to_{name_b}.csv', index=False)
    barchart(r2_ab, name_a, name_b, f'{out}/r2_barchart_{name_a}_to_{name_b}.png', top_n)
    corr_clustermap(A, B, name_a, name_b, out)

    summary = {
        'method_A': name_a, 'method_B': name_b,
        'n_fish': int(A.shape[0]), 'dims_A': int(A.shape[1]), 'dims_B': int(B.shape[1]),
        'classes': (np.bincount(labels.astype(int)).tolist() if labels is not None else None),
        'linear_CKA': round(cka, 3),
        'mean_cross_prediction_R2_A_to_B': round(float(np.mean(list(r2_ab.values()))), 3),
        'top_canonical_corr': [round(c, 3) for c in canon],
        'per_feature_R2_A_to_B': {k: round(v, 3) for k, v in sorted(r2_ab.items(), key=lambda kv: -kv[1])},
        'note': note,
    }
    if both:                                                # reverse direction B->A
        r2_ba = cross_pred_r2(Xb, pd.DataFrame(Xa, columns=A.columns), alpha)
        pd.DataFrame(sorted(r2_ba.items(), key=lambda kv: -kv[1]), columns=['feature', 'cv_r2']) \
            .to_csv(f'{out}/cross_prediction_r2_{name_b}_to_{name_a}.csv', index=False)
        barchart(r2_ba, name_b, name_a, f'{out}/r2_barchart_{name_b}_to_{name_a}.png', top_n)
        summary['mean_cross_prediction_R2_B_to_A'] = round(float(np.mean(list(r2_ba.values()))), 3)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-a', required=True, help='feature CSV A (predictor/source)')
    ap.add_argument('--feat-b', required=True, help='feature CSV B (target)')
    ap.add_argument('--name-a', default=None); ap.add_argument('--name-b', default=None)
    ap.add_argument('--id-col', default=None, help='override the data-id column (default: auto-detect)')
    ap.add_argument('--label-col', default='label')
    ap.add_argument('--alpha', type=float, default=10.0, help='ridge regularisation for cross-prediction')
    ap.add_argument('--top-n', type=int, default=30, help='max features shown in the R^2 bar chart')
    ap.add_argument('--both-directions', action='store_true', help='also compute B->A cross-prediction')
    ap.add_argument('--note', default='', help='free-text provenance recorded in the summary')
    ap.add_argument('--out', default=f'{os.path.dirname(os.path.abspath(__file__))}/results_representation_compare')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    name_a = args.name_a or os.path.splitext(os.path.basename(args.feat_a))[0]
    name_b = args.name_b or os.path.splitext(os.path.basename(args.feat_b))[0]
    A, la = load_features(args.feat_a, args.id_col, args.label_col)
    B, lb = load_features(args.feat_b, args.id_col, args.label_col)

    common = A.index.intersection(B.index)
    if len(common) < 10:
        raise SystemExit(f"[ERR] only {len(common)} shared samples between A and B — check the id columns match")
    if len(common) < len(A) or len(common) < len(B):
        print(f"[INFO] aligned on {len(common)} shared samples (A had {len(A)}, B had {len(B)})")
    A, B = A.loc[common], B.loc[common]
    labels = (la.loc[common] if la is not None else (lb.loc[common] if lb is not None else None))

    summary = analyse(name_a, name_b, A, B, args.out, args.alpha, args.top_n,
                      args.both_directions, args.note, labels)
    with open(f'{args.out}/comparison_summary.txt', 'w') as fh:
        fh.write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'\n[OK] saved -> {args.out}/  (comparison_summary.txt, cross_prediction_r2_*.csv, r2_barchart_*.png)')


if __name__ == '__main__':
    main()
