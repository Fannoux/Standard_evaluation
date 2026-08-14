#!/usr/bin/env python3
"""
Emit the LaTeX results table(s) from an all_methods_comparison.csv (standard_eval output).
Tweak the CONFIG block: which comparison CSV, which methods/tags, which VAE run, class prior.

  python make_results_table.py            # prints LaTeX to stdout
"""
import os
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(HERE, 'HOLDOUT_TestSet_Results', 'ECCV_frozen_TrainTest', 'all_methods_comparison.csv')

# display name -> tag in the CSV 'method' column, in the order to print
METHODS = [
    ('RegionProps',      'ECCV_FROZEN_RegionProps_Test'),
    ('CNN',              'ECCV_FROZEN_CNN_Test'),
    ('ShapeEmbed',       'ECCV_FROZEN_ShapeEmbed_Test'),
    ('VAE (spring)',     'ECCV_FROZEN_VAE_spring'),      # TODO: rename to the config that distinguishes them
    ('VAE (ethereal)',   'ECCV_FROZEN_VAE_ethereal'),    #       (e.g. VAE ($\beta$=..) / latent size)
]
CLASS_PRIOR = [366, 1009, 347, 110, 19]                 # SC0..SC4 test counts (for the majority baseline)
METRICS = [('acc', 'Acc'), ('macroF1', 'F1'), ('quadKappa', r'$\kappa$')]
BOLD_METRICS = {'macroF1', 'quadKappa'}                 # accuracy is NOT bolded (majority-baseline trap)
# ===================================================================


def majority_baseline():
    p = np.array(CLASS_PRIOR); N = p.sum(); maj = p.argmax()
    # 5-class: predict the majority class
    acc5 = p[maj] / N
    f1_maj = 2 * (p[maj]/N) / ((p[maj]/N) + 1)            # prec=p_maj/N, rec=1
    mf5 = f1_maj / len(p)
    # binary: no-kink (SC0) vs kink (SC1+); majority = kink
    nk = p[0]; k = p[1:].sum()
    accb = k / N
    f1_k = 2 * (k/N) / ((k/N) + 1)
    mfb = f1_k / 2
    return {'5class': (acc5, mf5, 0.0), 'binary': (accb, mfb, 0.0)}


def cell(v, best):
    s = f'{v:.2f}'
    return rf'\textbf{{{s}}}' if best else s


def build(df):
    rows = {}   # (disp, clf) -> {task: {metric: val}}
    dims = {}
    for disp, tag in METHODS:
        sub = df[df['method'] == tag]
        if sub.empty:
            print(f'% [WARN] no rows for {tag}')
            continue
        dims[disp] = int(sub['dims'].iloc[0])
        for _, r in sub.iterrows():
            rows.setdefault((disp, r['classifier']), {}).setdefault(r['task'], {})
            for m, _ in METRICS:
                rows[(disp, r['classifier'])][r['task']][m] = float(r[m])

    # best per (task, metric) column across all method/clf rows (exclude baseline)
    best = {}
    for task in ('5class', 'binary'):
        for m, _ in METRICS:
            if m not in BOLD_METRICS:
                best[(task, m)] = None
                continue
            vals = [d[task][m] for d in rows.values() if task in d and m in d[task]]
            best[(task, m)] = max(vals) if vals else None

    base = majority_baseline()
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r"\caption{Held-out test-set performance (1{,}851 F2 larvae). "
                 r"Acc = accuracy, F1 = macro-averaged F1, $\kappa$ = quadratic-weighted Cohen's $\kappa$. "
                 r"Best F1/$\kappa$ per column in \textbf{bold}; accuracy is left unbolded as it is "
                 r"dominated by the class prior (cf.\ the majority baseline).}")
    lines.append(r'\label{tab:test_results}')
    lines.append(r'\begin{tabular}{llc ccc ccc}')
    lines.append(r'\toprule')
    lines.append(r' & & & \multicolumn{3}{c}{5-class} & \multicolumn{3}{c}{Binary} \\')
    lines.append(r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}')
    lines.append(r'Representation & Clf & Dim & Acc & F1 & $\kappa$ & Acc & F1 & $\kappa$ \\')
    lines.append(r'\midrule')
    # majority baseline
    b5, bb = base['5class'], base['binary']
    lines.append(rf'Majority baseline & -- & -- & {b5[0]:.2f} & {b5[1]:.2f} & {b5[2]:.2f} '
                 rf'& {bb[0]:.2f} & {bb[1]:.2f} & {bb[2]:.2f} \\')
    lines.append(r'\midrule')
    for disp, _ in METHODS:
        for ci, clf in enumerate(('LogReg', 'RF')):
            if (disp, clf) not in rows:
                continue
            d = rows[(disp, clf)]
            name = rf'\multirow{{2}}{{*}}{{{disp}}}' if ci == 0 else ''
            dimc = rf'\multirow{{2}}{{*}}{{{dims[disp]}}}' if ci == 0 else ''
            c5 = [cell(d['5class'][m], d['5class'][m] == best[('5class', m)]) for m, _ in METRICS]
            cb = [cell(d['binary'][m], d['binary'][m] == best[('binary', m)]) for m, _ in METRICS]
            lines.append(rf'{name} & {clf} & {dimc} & ' + ' & '.join(c5) + ' & ' + ' & '.join(cb) + r' \\')
        lines.append(r'\addlinespace[2pt]')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


if __name__ == '__main__':
    df = pd.read_csv(TEST_CSV)
    print(build(df))
