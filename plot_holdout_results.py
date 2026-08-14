#!/usr/bin/env python3
"""
Plot holdout test-set results: confusion matrices, per-class recall, method comparison.
pandas + seaborn + matplotlib. Everything is driven by the CONFIG block below -- tweak freely.

Inputs (already produced by standard_eval), all under RESULTS_DIR:
  - all_methods_comparison.csv                              (metrics per method/task/classifier)
  - confusion_ECCV_FROZEN_<tag>_Test_<task>_<clf>.csv       (raw NxN counts, no header)

Run:  python plot_holdout_results.py      -> writes PNGs into RESULTS_DIR/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== CONFIG (tweak here) ==============================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'HOLDOUT_TestSet_Results', 'ECCV_frozen_TrainTest')
COMPARISON_CSV = os.path.join(RESULTS_DIR, 'all_methods_comparison.csv')
OUT_DIR = os.path.join(RESULTS_DIR, 'figures')

# display name -> the FULL <tag> in the confusion filename confusion_ECCV_FROZEN_<tag>_<task>_<clf>.csv
# (note: CNN/ShapeEmbed/RegionProps carry a _Test suffix, the two VAE runs do not)
METHODS = {
    'CNN':            'CNN_Test',
    'ShapeEmbed':     'ShapeEmbed_Test',
    'RegionProps':    'RegionProps_Test',
    'VAE (spring)':   'VAE_spring',
    'VAE (ethereal)': 'VAE_ethereal',
}
CLASSIFIER = 'RF'                                   # 'RF' or 'LogReg'
CLASS_NAMES  = ['SC0', 'SC1', 'SC2', 'SC3', 'SC4']  # 5-class labels
BINARY_NAMES = ['no kink', 'kink']                  # binary labels
NORMALISE = 'true'                                  # 'true' = row-normalise (recall); None = raw counts
PALETTE = 'colorblind'
sns.set_theme(context='paper', style='whitegrid', palette=PALETTE)
# try: import scienceplots; plt.style.use(['science','no-latex'])   # optional, if installed
# ================================================================================


def confusion_path(tag, task, clf=CLASSIFIER):
    return os.path.join(RESULTS_DIR, f'confusion_ECCV_FROZEN_{tag}_{task}_{clf}.csv')


def load_confusion(tag, task='5class', clf=CLASSIFIER):
    """Raw NxN count matrix (rows=true, cols=pred)."""
    return pd.read_csv(confusion_path(tag, task, clf), header=None).values.astype(float)


def names_for(task):
    return BINARY_NAMES if task == 'binary' else CLASS_NAMES


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name + '.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print('[OK] ->', p)


# ------------------------------- plots -------------------------------
def plot_confusions(task='5class', clf=CLASSIFIER):
    names = names_for(task)
    n = len(METHODS)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, (disp, tag) in zip(axes[0], METHODS.items()):
        cm = load_confusion(tag, task, clf)
        M = cm / cm.sum(1, keepdims=True) if NORMALISE == 'true' else cm
        sns.heatmap(M, annot=True, fmt='.2f' if NORMALISE else '.0f', cmap='Blues',
                    vmin=0, vmax=(1 if NORMALISE else None), cbar=False,
                    xticklabels=names, yticklabels=names, ax=ax, annot_kws={'size': 8})
        ax.set_title(disp)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true' if ax is axes[0][0] else '')
    kind = 'recall (row-normalised)' if NORMALISE == 'true' else 'counts'
    fig.suptitle(f'Holdout test confusion — {task}, {clf} — {kind}', y=1.03)
    fig.tight_layout()
    save(fig, f'confusion_{task}_{clf}')


def per_class_recall_df(task='5class', clf=CLASSIFIER):
    names = names_for(task)
    rows = []
    for disp, tag in METHODS.items():
        cm = load_confusion(tag, task, clf)
        rec = cm.diagonal() / cm.sum(1)
        rows += [{'method': disp, 'class': c, 'recall': r} for c, r in zip(names, rec)]
    return pd.DataFrame(rows)


def plot_per_class_recall(task='5class', clf=CLASSIFIER):
    df = per_class_recall_df(task, clf)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.barplot(df, x='class', y='recall', hue='method', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f'Per-class recall — {task}, {clf}')
    ax.legend(title='', frameon=False)
    fig.tight_layout()
    save(fig, f'per_class_recall_{task}_{clf}')


def plot_metric_comparison(metrics=('quadKappa', 'macroF1', 'acc'), task='5class', clf=CLASSIFIER):
    df = pd.read_csv(COMPARISON_CSV)
    df = df[(df['task'] == task) & (df['classifier'] == clf)].copy()
    df['method'] = (df['method'].str.replace('ECCV_FROZEN_', '', regex=False)
                                .str.replace('_Test', '', regex=False))
    long = df.melt(id_vars='method', value_vars=list(metrics),
                   var_name='metric', value_name='score')
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.barplot(long, x='metric', y='score', hue='method', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f'Method comparison — {task}, {clf}')
    ax.legend(title='', frameon=False)
    # annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', fontsize=7, padding=2)
    fig.tight_layout()
    save(fig, f'metric_comparison_{task}_{clf}')


if __name__ == '__main__':
    plot_confusions('5class')
    plot_confusions('binary')
    plot_per_class_recall('5class')
    plot_metric_comparison(('quadKappa', 'macroF1', 'acc'), '5class')
    plot_metric_comparison(('quadKappa', 'macroF1', 'acc'), 'binary')
    # plt.show()   # uncomment to view interactively instead of just saving
    print('done.')
