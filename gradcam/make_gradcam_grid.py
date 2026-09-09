#!/usr/bin/env python3
"""
Assemble a figure grid from run_gradcam outputs: rows = label, cols = methods.

Exemplar per row = the sample (present for all methods) with the highest body-third enrichment among
well-predicted samples. By default tiles the saved overlay PNGs; --rerender redraws image panels from
cams/*.npy with top-mass threshold + outline (no GPU/model needed).

Usage:
  python make_gradcam_grid.py --results results_gradcam --methods CNN ShapeEmbed VAE \
      --titles "CNN" "ShapeEmbed" "VAE" --rerender --csv <manifest> --mask-dir <masks> --mask-ext .png \
      --out gradcam_grid.png
"""
import argparse
import glob
import os
import re

import numpy as np


# --------------------------- display helpers (self-contained) ---------------------------
def _minmax(im):
    im = np.asarray(im, float)
    lo, hi = im.min(), im.max()
    return (im - lo) / (hi - lo + 1e-8)


def top_mass(cam, frac=0.2):
    """Keep only pixels carrying the top `frac` of total CAM mass; rest -> NaN (transparent)."""
    if not frac or frac >= 1:
        return cam
    flat = np.asarray(cam, float).ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]); total = csum[-1] + 1e-12
    k = int(np.searchsorted(csum, frac * total)) + 1
    thr = flat[order][min(k, len(flat) - 1)]
    return np.where(np.asarray(cam) >= thr, cam, np.nan)


def _load_gray(path, size):
    from skimage.io import imread
    from skimage.transform import resize
    im = imread(path).astype(np.float32)
    if im.ndim == 3:
        im = im.mean(-1)
    return resize(im, (size, size), order=1, preserve_range=True, anti_aliasing=True)


def _load_mask(mask_dir, stem, ext, size):
    from skimage.io import imread
    from skimage.transform import resize
    mp = os.path.join(mask_dir, stem + ext)
    m = np.load(mp) if mp.endswith('.npy') else imread(mp)
    m = np.asarray(m); m = m[..., 0] if m.ndim == 3 else m
    return resize((m > 0).astype(float), (size, size), order=0, preserve_range=True)


# --------------------------- indexing + exemplar choice ---------------------------
def index_overlays(overlay_dir, methods):
    idx = {}
    for m in methods:
        for f in glob.glob(os.path.join(overlay_dir, f'{m}_SC*_*.png')):
            base = os.path.basename(f)[len(m) + 1:]
            mm = re.match(r'SC(\d+)_(.+)\.png$', base)
            if mm:
                idx.setdefault((m, int(mm.group(1))), {})[mm.group(2)] = f
    return idx


def choose_exemplars(results_dir, methods, scs, idx):
    """Per row: among samples with overlays for all methods, pick highest body enrichment,
    tie-broken by smallest mean |E[k]-label|."""
    enr, err = {}, {}
    pf = os.path.join(results_dir, 'gradcam_perfish.csv')
    if os.path.exists(pf):
        import pandas as pd
        df = pd.read_csv(pf)
        ecols = [c for c in ('enr_body_ant', 'enr_body_mid', 'enr_body_post') if c in df.columns]
        for _, r in df.iterrows():
            key = (int(r['SC']), str(r['fish']))
            if 'Ek' in df.columns and not pd.isna(r['Ek']):
                err.setdefault(key, []).append(abs(float(r['Ek']) - int(r['SC'])))
            if ecols:
                vals = [r[c] for c in ecols if not pd.isna(r[c])]
                if vals:
                    enr[key] = max(enr.get(key, 0.0), float(np.mean(vals)))   # best image method's body enrichment
    chosen = {}
    for sc in scs:
        common = None
        for m in methods:
            common = set(idx.get((m, sc), {})) if common is None else (common & set(idx.get((m, sc), {})))
        common = sorted(common or [])
        if not common:
            chosen[sc] = None
            print(f"[warn] SC{sc}: no sample has panels for all methods -> row blank")
            continue
        chosen[sc] = max(common, key=lambda fid: (enr.get((sc, fid), 0.0),
                                                  -np.mean(err.get((sc, fid), [9.9]))))
    return chosen


# --------------------------- rendering ---------------------------
def render_cell(ax, method, sc, fid, png_path, args, plt):
    """Re-render an image-method panel from the raw CAM (threshold + outline) if possible; else PNG."""
    cam_npy = os.path.join(args.results, 'cams', f'{method}_{fid}.npy')
    img_path = args._paths.get(fid) if args.rerender else None
    if args.rerender and img_path and os.path.exists(cam_npy) and os.path.exists(img_path):
        try:
            cam = np.load(cam_npy)
            ax.imshow(_minmax(_load_gray(img_path, cam.shape[0])), cmap='gray')
            ax.imshow(top_mass(cam, args.top_frac), cmap='jet', alpha=0.5, vmin=0, vmax=1)
            if args.mask_dir:
                try:
                    m = _load_mask(args.mask_dir, fid, args.mask_ext, cam.shape[0])
                    if m.sum() > 0:
                        ax.contour(m, levels=[0.5], colors='cyan', linewidths=0.7)
                except Exception:
                    pass
            return
        except Exception as e:
            print(f"[warn] re-render {method}/{fid} failed ({e}); using saved PNG")
    if png_path and os.path.exists(png_path):
        ax.imshow(plt.imread(png_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results_gradcam')
    ap.add_argument('--methods', nargs='+', default=['CNN', 'ShapeEmbed'])
    ap.add_argument('--scs', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--titles', nargs='+')
    ap.add_argument('--out', default='gradcam_grid.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--rerender', action='store_true',
                    help='re-render image panels from cams/*.npy with top-mass threshold + outline')
    ap.add_argument('--top-frac', type=float, default=0.2, help='fraction of CAM mass to display')
    ap.add_argument('--csv', help='manifest (for --rerender: data_id -> image_path)')
    ap.add_argument('--mask-dir'); ap.add_argument('--mask-ext', default='.png')
    a = ap.parse_args()
    titles = a.titles or a.methods
    if len(titles) != len(a.methods):
        raise SystemExit('[ERR] --titles must have one entry per method')

    # data_id -> image_path for re-render
    a._paths = {}
    if a.rerender and a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv)
        stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
        a._paths = {stem(p): p for p in df['image_path']}

    idx = index_overlays(os.path.join(a.results, 'overlays'), a.methods)
    chosen = choose_exemplars(a.results, a.methods, a.scs, idx)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nr, nc = len(a.scs), len(a.methods)
    fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.4 * nr), squeeze=False)
    for ri, sc in enumerate(a.scs):
        fid = chosen[sc]
        for ci, m in enumerate(a.methods):
            ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(f'SC{sc}', fontsize=11, rotation=90, labelpad=8)
            if fid is not None:
                render_cell(ax, m, sc, fid, idx.get((m, sc), {}).get(fid), a, plt)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches='tight')
    plt.close(fig)
    picks = ", ".join(f"SC{sc}:{chosen[sc]}" for sc in a.scs)
    print(f"[OK] -> {a.out}  ({nr}x{nc}; {'re-rendered' if a.rerender else 'tiled PNGs'}; exemplars -> {picks})")


if __name__ == '__main__':
    main()
