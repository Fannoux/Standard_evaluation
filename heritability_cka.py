#!/usr/bin/env python3
r"""
Heritability alignment of representations: how much each representation's geometry reflects
GENETIC relatedness. Dimension-invariant -> CNN (512-d) and RegionProps (19-d) compare fairly.

Per method:  CKA(X Xᵀ, GRM)  [headline, 0..1]  and off-diagonal pair-correlation [HE flavour].
Reads GCTA/PLINK native files (binary GRM + .pheno_formatted + gcta.cov). Fish are matched on IID
(2nd column) — the sequencing_id — which is shared across all of the files.

USAGE (on the cluster, from the genetics input dir):
  python heritability_cka.py \
    --relmat-bin input.grm.bin --relmat-id input.grm.id \
    --covariates gcta.cov \
    --feats CNN=CNN_gcta.pheno_formatted_norm \
            RegionProps=RegionProps_gcta.pheno_formatted_norm \
            ShapeEmbed=shapeembed_gcta.pheno_formatted_norm \
            VAE_spring=vae_spring-sweep-14_gcta.pheno_formatted_norm \
            VAE_ethereal=vae_ethereal-sweep-20_gcta.pheno_formatted_norm \
            VAE_worthy=vae_worthy-sweep-27_gcta.pheno_formatted_norm \
            adjSC=adjSC_gcta.pheno_formatted_norm \
    --out results_heritability

Notes:
  * adjSC is the severity phenotype itself -> its alignment is the reference "how heritable is the trait".
  * --relmat-bin auto-detects GCTA (.grm.bin float32 lower-tri) vs PLINK (.rel.bin square/tri, f32/f64).
  * --covariates gcta.cov (FID IID c1 c2 ...) are residualised out first so alignment reflects genetics,
    not shared environment. Drop the flag to skip (it will warn).
  * Feature files may also be plain CSVs (fish_id, f0..fN, label) — auto-detected.
"""
import argparse, os
import numpy as np
import pandas as pd


# ------------------------- core measures -------------------------
def cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    d = np.linalg.norm(Kc) * np.linalg.norm(Lc)
    return float((Kc * Lc).sum() / d) if d else float('nan')


def offdiag_corr(K, G):
    iu = np.triu_indices(K.shape[0], k=1)
    return float(np.corrcoef(K[iu], G[iu])[0, 1])


# ------------------------- readers -------------------------
def read_relmat_bin(bin_path, id_path):
    """GCTA .grm.bin (f32 lower-tri) or PLINK .rel.bin (square/tri, f32/f64) -> DataFrame indexed by IID."""
    ids = [ln.split()[1] for ln in open(id_path) if ln.strip()]      # 2nd col = IID = sequencing id
    n = len(ids); nb = os.path.getsize(bin_path)
    ntri, nsq = n * (n + 1) // 2, n * n
    for dtype, s in ((np.float32, 4), (np.float64, 8)):
        if nb == ntri * s:
            v = np.fromfile(bin_path, dtype=dtype); M = np.zeros((n, n))
            M[np.tril_indices(n)] = v; M = M + M.T - np.diag(np.diag(M))
            return pd.DataFrame(M, index=ids, columns=ids)
        if nb == nsq * s:
            return pd.DataFrame(np.fromfile(bin_path, dtype=dtype).reshape(n, n), index=ids, columns=ids)
    raise SystemExit(f"{bin_path}: {nb} bytes matches neither triangle nor square for n={n} ids")


def load_grm(a):
    if a.relmat_bin:
        return read_relmat_bin(a.relmat_bin, a.relmat_id)
    if a.genotypes:
        g = pd.read_csv(a.genotypes, index_col=0); M = g.values.astype(float)
        p = np.nanmean(M, 0) / 2.0
        Z = np.nan_to_num((M - 2 * p) / np.sqrt(2 * p * (1 - p) + 1e-12))
        return pd.DataFrame((Z @ Z.T) / Z.shape[1], index=g.index.astype(str), columns=g.index.astype(str))
    grm = pd.read_csv(a.grm, index_col=0)
    grm.index = grm.index.astype(str); grm.columns = grm.columns.astype(str)
    return grm


def load_feats(path):
    """CSV (has 'fish_id') OR GCTA .pheno (whitespace, no header: FID IID p1 p2 ...). -> ids, X."""
    if 'fish_id' in open(path).readline():
        df = pd.read_csv(path)
        ids = df['fish_id'].astype(str).values
        X = df.drop(columns=[c for c in ('fish_id', 'label') if c in df.columns]).select_dtypes('number')
    else:
        df = pd.read_csv(path, sep=r'\s+', header=None)
        ids = df[1].astype(str).values                                # IID
        X = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    X = X.replace(-9, np.nan)                                          # GCTA missing code
    return ids, X.fillna(X.mean()).values


def load_cov(path):
    """gcta.cov (FID IID ...) or CSV (index=fish_id). -> DataFrame indexed by id."""
    if ',' in open(path).readline():
        return pd.read_csv(path, index_col=0)
    df = pd.read_csv(path, sep=r'\s+', header=None)
    return df.iloc[:, 2:].set_axis(df[1].astype(str).values)


def residualise(X, cov_df, ids):
    C = pd.get_dummies(cov_df.reindex(ids), dummy_na=True, drop_first=True).astype(float).fillna(0).values
    C = np.hstack([np.ones((len(C), 1)), C])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', nargs='+', required=True, help='name=path ... (.pheno_formatted or CSV)')
    ap.add_argument('--relmat-bin'); ap.add_argument('--relmat-id')
    ap.add_argument('--grm'); ap.add_argument('--genotypes')
    ap.add_argument('--covariates')
    ap.add_argument('--out', default='results_heritability')
    a = ap.parse_args()
    if a.relmat_bin and not a.relmat_id:
        ap.error('--relmat-bin needs --relmat-id')
    if not (a.relmat_bin or a.grm or a.genotypes):
        ap.error('provide --relmat-bin+--relmat-id, or --grm, or --genotypes')

    grm = load_grm(a)
    cov = load_cov(a.covariates) if a.covariates else None
    if cov is None:
        print('[WARN] no --covariates: alignment may absorb shared-environment (plate/tank) structure')

    rows = []
    for spec in a.feats:
        name, path = spec.split('=', 1)
        ids, X = load_feats(path)
        keep = np.array([i in grm.index for i in ids])
        if keep.sum() < 3:
            print(f'[WARN] {name}: {keep.sum()} fish matched the GRM -> skipped'); continue
        X, ids = X[keep], ids[keep]
        if cov is not None:
            X = residualise(X, cov, ids)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        G = grm.loc[ids, ids].values
        K = Xs @ Xs.T
        r = {'method': name, 'n_fish': int(keep.sum()), 'dims': X.shape[1],
             'cka_grm': round(cka(K, G), 3), 'corr_grm': round(offdiag_corr(K, G), 3)}
        rows.append(r)
        print(f"[{name}] n={r['n_fish']} dims={r['dims']}  CKA(rep,GRM)={r['cka_grm']}  pair-corr={r['corr_grm']}")

    os.makedirs(a.out, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, 'heritability_alignment.csv'), index=False)
    print('\n' + res.to_string(index=False))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(res['method'], res['cka_grm']); ax.set_ylabel('CKA(representation, GRM)')
        ax.set_ylim(0, max(0.1, res['cka_grm'].max() * 1.25))
        ax.set_title('Heritability alignment (higher = more genetically structured)')
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(res['cka_grm']):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'heritability_cka_barchart.png'), dpi=300)
        print(f"[OK] -> {a.out}/heritability_cka_barchart.png")
    except Exception as e:
        print(f'[WARN] plot skipped: {e}')


if __name__ == '__main__':
    main()
