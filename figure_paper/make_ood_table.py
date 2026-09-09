#!/usr/bin/env python3
"""
Emit the LaTeX validation-vs-test (OOD gap) table.
Default RF, 5-class: Val vs Test macro-F1 and quad-kappa, plus Delta-kappa.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(HERE, 'HOLDOUT_TestSet_Results', 'ECCV_frozen_TrainTest', 'all_methods_comparison.csv')
# TrainVal holds the full validation comparison
VAL_CSV = os.path.join(HERE, 'TrainVal_Results', 'ECCV_frozen_TrainVal', 'all_methods_comparison.csv')
CLAS_VAL = VAE_VAL = VAL_CSV

PROBE, TASK = 'RF', '5class'

# disp, dim, test_tag, (val_csv, val_tag)
METHODS = [
    ('RegionProps',    19,  'ECCV_FROZEN_RegionProps_Test', CLAS_VAL, 'ECCV_FROZEN_RegionProps'),
    ('CNN',            512, 'ECCV_FROZEN_CNN_Test',         CLAS_VAL, 'ECCV_FROZEN_CNN_best312'),
    ('ShapeEmbed',     128, 'ECCV_FROZEN_ShapeEmbed_Test',  CLAS_VAL, 'ECCV_FROZEN_ShapeEmbed'),
    ('VAE ',           256, 'ECCV_FROZEN_VAE',              VAE_VAL,  'ECCV_FROZEN_VAE'),
]

_cache = {}
def get(csv, tag):
    if csv not in _cache:
        _cache[csv] = pd.read_csv(csv)
    d = _cache[csv]
    r = d[(d['method'] == tag) & (d['task'] == TASK) & (d['classifier'] == PROBE)]
    if r.empty:
        raise SystemExit(f"[ERR] no {tag}/{TASK}/{PROBE} in {csv}")
    return float(r['macroF1'].iloc[0]), float(r['quadKappa'].iloc[0])


test_df = pd.read_csv(TEST_CSV)
rows = []
best_testk = -9
for disp, dim, ttag, vcsv, vtag in METHODS:
    r = test_df[(test_df['method'] == ttag) & (test_df['task'] == TASK) & (test_df['classifier'] == PROBE)]
    tF1, tK = float(r['macroF1'].iloc[0]), float(r['quadKappa'].iloc[0])
    vF1, vK = get(vcsv, vtag)
    rows.append((disp, dim, vF1, tF1, vK, tK, tK - vK))
    best_testk = max(best_testk, tK)

L = []
L.append(r'\begin{table}[t]')
L.append(r'\centering')
L.append(rf"\caption{{Validation vs.\ held-out test performance ({PROBE} probe, 5-class). "
         r"Validation is in-distribution (mixed F0/F2); the test set is entirely F2 (a later "
         r"generation). All representations degrade sharply out-of-distribution; the CNN is the most "
         r"robust. $\Delta\kappa$ = test $-$ validation.}")
L.append(r'\label{tab:ood_gap}')
L.append(r'\begin{tabular}{lc cc cc c}')
L.append(r'\toprule')
L.append(r' & & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{$\kappa$} & \\')
L.append(r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}')
L.append(r'Representation & Dim & Val & Test & Val & Test & $\Delta\kappa$ \\')
L.append(r'\midrule')
for disp, dim, vF1, tF1, vK, tK, dK in rows:
    tkc = rf'\textbf{{{tK:.2f}}}' if tK == best_testk else f'{tK:.2f}'
    L.append(rf'{disp} & {dim} & {vF1:.2f} & {tF1:.2f} & {vK:.2f} & {tkc} & {dK:+.2f} \\')
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
print('\n'.join(L))
