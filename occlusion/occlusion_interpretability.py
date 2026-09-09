#!/usr/bin/env python3
"""Occlusion-based interpretability, unified across representations.

Blank a region of the input, re-extract, re-predict through the same probe, and measure the
prediction shift. Works for any method via a forward pass; __main__ runs a smoke test.
"""
import os
import numpy as np
import pandas as pd


# ------------------------- region definition (on the mask) -------------------------
def define_regions(mask):
    """mask: 2-D array, >0 = sample. Returns {region_name: boolean pixel-mask}."""
    fg = mask > 0
    reg = {'background': ~fg, 'foreground': fg}
    ys, xs = np.where(fg)
    if len(xs) >= 3:
        pts = np.stack([xs, ys]).astype(float).T
        c = pts - pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov(c.T))          # major axis = last eigenvector
        proj = c @ vecs[:, -1]
        q1, q2 = np.quantile(proj, [1/3, 2/3])
        for name, sel in [('body_ant', proj <= q1),
                          ('body_mid', (proj > q1) & (proj <= q2)),
                          ('body_post', proj > q2)]:
            m = np.zeros(mask.shape, bool)
            m[ys[sel], xs[sel]] = True
            reg[name] = m
    return reg


def perturb(image, mask, region, fill=0.0):
    """Blank `region` in both image and mask."""
    im, mk = image.copy(), mask.copy()
    im[region] = fill
    mk[region] = 0
    return im, mk


# ------------------------- the unified occlusion loop -------------------------
def occlusion_sensitivity(images, masks, extract, predict,
                          regions=('background', 'foreground', 'body_ant', 'body_mid', 'body_post')):
    """Return (per-sample long df, per-region mean |Δprediction|)."""
    rows = []
    for i, (img, msk) in enumerate(zip(images, masks)):
        base = float(predict(extract(img, msk)))
        regs = define_regions(msk)
        for rn in regions:
            if rn not in regs or regs[rn].sum() == 0:
                continue
            im2, mk2 = perturb(img, msk, regs[rn])
            delta = abs(float(predict(extract(im2, mk2))) - base)
            rows.append({'fish': i, 'region': rn, 'delta': delta})
    df = pd.DataFrame(rows)
    summ = df.groupby('region')['delta'].mean()
    return df, summ


def sliding_window_map(image, mask, extract, predict, patch=32, stride=16, fill=0.0):
    """Grad-CAM-like heatmap for one image-based sample: Δprediction as a patch slides over it."""
    H, W = image.shape
    base = float(predict(extract(image, mask)))
    heat = np.zeros(((H - patch)//stride + 1, (W - patch)//stride + 1))
    for a, y in enumerate(range(0, H - patch + 1, stride)):
        for b, x in enumerate(range(0, W - patch + 1, stride)):
            im2 = image.copy(); im2[y:y+patch, x:x+patch] = fill
            heat[a, b] = abs(float(predict(extract(im2, mask))) - base)
    return heat


# ------------------------- plug-in points (REPLACE for real runs) -------------------------
def make_extractor(method):
    if method == 'dummy':                              # toy: label ~ elongation + fg brightness
        def extract(image, mask):
            fg = image[mask > 0]
            ys, xs = np.where(mask > 0)
            asp = (np.ptp(xs) + 1) / (np.ptp(ys) + 1) if len(xs) else 1.0
            return np.array([fg.mean() if fg.size else 0.0, mask.sum(), asp])
        return extract
    raise NotImplementedError(f"the smoke test only implements 'dummy'; real {method} extractors live in real_extractors.py")


def make_predictor(method):
    if method == 'dummy':                              # fixed linear read-out
        w = np.array([2.0, 0.0, 3.0])                  # brightness + elongation drive the label
        return lambda f: float(f @ w)
    raise NotImplementedError(f"the smoke test only implements 'dummy'; real {method} predictors live in real_extractors.py")


def load_data(method, n=8):
    """SMOKE-TEST synthetic data. For real runs use load_data_real() below."""
    rng = np.random.RandomState(0)
    imgs, msks = [], []
    for k in range(n):
        img = rng.rand(96, 96) * 0.2                   # noisy background
        msk = np.zeros((96, 96), np.uint8)
        msk[40:56, 20:20 + 40 + 4*k] = 1               # elongated sample of varying length
        img[msk > 0] += 0.6                            # sample is brighter
        img[10:18, 70:78] += 0.9                       # bright artifact in the background
        imgs.append(img.astype(np.float32)); msks.append(msk)
    return imgs, msks


def load_data_real(csv, image_col='image_path', label_col='severity_score_adjusted',
                   split='test', split_col='set', mask_dir=None, mask_col=None, mask_ext='.npy',
                   n_per_class=40, work_size=512, seed=0, subset_file=None):
    """Read the manifest CSV -> a stratified subset of paired (image, mask)."""
    import pandas as pd
    from skimage.io import imread
    from skimage.transform import resize
    df = pd.read_csv(csv)
    if split is not None and split_col in df.columns:
        df = df[df[split_col] == split]
    stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    if subset_file and os.path.exists(subset_file):
        ids = {l.strip() for l in open(subset_file) if l.strip()}
        sub = df[df[image_col].map(stem).isin(ids)].reset_index(drop=True)
        print(f"[load] reusing frozen subset {subset_file} ({len(sub)} samples)")
    else:
        sub = pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed) # stratified by class
                         for _, g in df.groupby(label_col)]).reset_index(drop=True)
        if subset_file:
            open(subset_file, 'w').write('\n'.join(sub[image_col].map(stem)) + '\n')
            print(f"[load] froze subset -> {subset_file}")
    def load_img(p):
        im = imread(p).astype(np.float32)
        if im.ndim == 3: im = im.mean(-1)
        return resize(im, (work_size, work_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    def load_mask(row):
        mp = row[mask_col] if (mask_col and pd.notna(row.get(mask_col))) \
             else os.path.join(mask_dir, stem(row[image_col]) + mask_ext)
        m = np.load(mp) if str(mp).endswith('.npy') else imread(mp)
        m = np.asarray(m); m = m[..., 0] if m.ndim == 3 else m
        return resize((m > 0).astype(float), (work_size, work_size), order=0, preserve_range=True).astype(np.uint8)

    imgs, msks, kept = [], [], []
    for _, row in sub.iterrows():
        try:
            imgs.append(load_img(row[image_col])); msks.append(load_mask(row)); kept.append(row)
        except Exception as e:
            print(f'[skip] {row[image_col]}: {e}')
    meta = pd.DataFrame(kept).reset_index(drop=True)
    meta['data_id'] = meta[image_col].map(stem)
    print(f"[load] {len(imgs)} samples ({split}), stratified by {label_col} "
          f"({sub[label_col].value_counts().sort_index().to_dict()})")
    return imgs, msks, meta


# ------------------------- driver -------------------------
def run(method, imgs=None, msks=None, extract=None, predict=None, out='results_occlusion'):
    if imgs is None:                                    # fall back to synthetic smoke-test data
        imgs, msks = load_data(method)
    if extract is None: extract = make_extractor(method)
    if predict is None: predict = make_predictor(method)
    df, summ = occlusion_sensitivity(imgs, msks, extract, predict)
    bg = summ.get('background', 0.0); fg = summ.get('foreground', 0.0)
    frac = bg / (bg + fg) if (bg + fg) else float('nan')
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f'occlusion_{method}.csv'), index=False)
    print(f"\n[{method}] mean |Δprediction| per region:\n{summ.round(4).to_string()}")
    print(f"[{method}] background sensitivity fraction = {frac:.3f}  "
          f"(near 0 = biology-driven, high = artifact-prone)")
    return summ, frac


if __name__ == '__main__':
    # smoke test on synthetic data with the dummy extractor/predictor
    run('dummy')
