#!/usr/bin/env python3
"""
CLI driver for the Grad-CAM cross-architecture visualization.

Loads the same frozen fish subset as the occlusion analysis, fits a matched linear severity head per
method, and for each fish computes:
  * CNN / VAE  -> image Grad-CAM on layer4, saved as an overlay PNG + raw .npy
  * ShapeEmbed -> contour-point saliency, saved as a colored-outline PNG
Plus consistency metrics: mask containment, per-body-third CAM mass, and CNN-vs-VAE spatial agreement.

Reuses the occlusion folder's loaders so both analyses stay on identical fish / identical model wiring.

Example (see run_gradcam.slurm):
  python run_gradcam.py --csv manifest.csv --mask-dir masks --mask-ext .png \
      --methods CNN VAE ShapeEmbed --cnn-weights cnn.pth --vae-weights vae.pt --shapeembed-weights se.pth \
      --probe-train CNN=cnn_features_train.csv VAE=vae_features_train.csv ShapeEmbed=se_features_train.csv \
      --out results_gradcam
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# --- reuse the occlusion folder (loaders, data, regions, preprocessing helpers) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_OCC = os.path.abspath(os.path.join(_HERE, '..', 'occlusion'))
if _OCC not in sys.path:
    sys.path.insert(0, _OCC)
from run_occlusion import load_cnn, load_vae, load_shapeembed          # noqa: E402
from occlusion_interpretability import load_data_real, define_regions  # noqa: E402
from real_extractors import _minmax, _resize, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from gradcam_engine import (fit_torch_head, fit_head_from_array, image_gradcam, contour_saliency,  # noqa: E402
                            expected_severity)


def _device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


# --------------------------- preprocessing (mirrors the occlusion extractors) ---------------------------
def prep_cnn(image, size, mean, std):
    import torch
    x = _resize(_minmax(image), size)
    x = np.repeat(x[None], 3, axis=0)
    x = (x - np.asarray(mean, float)[:, None, None]) / np.asarray(std, float)[:, None, None]
    return torch.tensor(x[None], dtype=torch.float32)


def prep_vae(image, size):
    import torch
    x = _resize(image.astype(np.float32), size)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return torch.tensor(x[None, None], dtype=torch.float32)


def _center_crop_resize(arr, size, crop, order=1):
    """A.CenterCrop(crop) -> A.Resize(size), matching the VAE transform (order=0 for masks)."""
    from skimage.transform import resize
    H, W = arr.shape
    ch, cw = min(crop, H), min(crop, W)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    arr = arr[y0:y0 + ch, x0:x0 + cw]
    return resize(arr, (size, size), order=order, preserve_range=True, anti_aliasing=(order == 1))


def crop_gray(image_path, size=352, crop=1536):
    """Full-res image -> center-crop -> resize (grayscale display array, NOT standardized)."""
    from skimage.io import imread
    im = imread(image_path).astype(np.float32)
    if im.ndim == 3:
        im = im.mean(-1)
    return _center_crop_resize(im, size, crop, order=1).astype(np.float32)


def crop_mask(mask_path, size=352, crop=1536, ref_shape=None):
    """Full-res mask -> same center-crop -> resize (nearest). Returns None if the mask's native shape
    does not match the image's (ref_shape) -- then the identical crop would not align, so no outline."""
    from skimage.io import imread
    m = np.load(mask_path) if str(mask_path).endswith('.npy') else imread(mask_path)
    m = np.asarray(m); m = m[..., 0] if m.ndim == 3 else m
    if ref_shape is not None and m.shape[:2] != tuple(ref_shape):
        return None
    return _center_crop_resize((m > 0).astype(float), size, crop, order=0).astype(np.uint8)


def place_crop_in_full(cam_crop, native_shape, out_hw, crop=1536):
    """Map a CAM computed on the central `crop` of a native-size image back into the full frame at
    out_hw, at the crop's true footprint (zero outside -- the VAE never saw the periphery). This puts
    the VAE panel in the SAME frame and scale as the CNN, so the grid is visually consistent and the
    containment / cross-backbone metrics are computed against the same 512-frame mask."""
    from skimage.transform import resize
    H0, W0 = native_shape
    ch, cw = min(crop, H0), min(crop, W0)
    oh, ow = out_hw
    bh, bw = max(1, round(ch / H0 * oh)), max(1, round(cw / W0 * ow))
    y0, x0 = (oh - bh) // 2, (ow - bw) // 2
    full = np.zeros((oh, ow), np.float32)
    full[y0:y0 + bh, x0:x0 + bw] = resize(np.asarray(cam_crop, float), (bh, bw), order=1, preserve_range=True)
    return full


def prep_vae_fullres(image_path, size=352, crop=1536):
    """The VAE inference transform (src_vae/data.py): read the full-res image ->
    A.CenterCrop(crop) -> A.Resize(size) -> per-image standardization (A.Normalize(normalization=
    'image')). The whole-frame downsample used elsewhere loses both the field of view (the central
    crop) and the resolution the encoder trained on; path-based so we can read full resolution."""
    import torch
    im = crop_gray(image_path, size, crop)
    im = (im - im.mean()) / (im.std() + 1e-8)                      # A.Normalize(normalization='image')
    return torch.tensor(im[None, None], dtype=torch.float32)


def load_split_for_fit(csv, label_col, split, split_col, mask_dir, mask_col, mask_ext,
                       n_per_class, work_size, seed, image_col='image_path'):
    """Self-contained loader for head fitting (keeps this inside the gradcam folder -- no dependency on
    the occlusion module's signature). Loads images always; masks best-effort (zero mask if missing,
    since CNN/VAE ignore the mask and only ShapeEmbed needs it). Returns (imgs, masks, labels[int])."""
    import pandas as pd
    from skimage.io import imread
    from skimage.transform import resize
    df = pd.read_csv(csv)
    if split is not None and split_col in df.columns:
        df = df[df[split_col] == split]
    stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    sub = pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed)
                     for _, g in df.groupby(label_col)]).reset_index(drop=True)

    def load_img(p):
        im = imread(p).astype(np.float32)
        if im.ndim == 3:
            im = im.mean(-1)
        return resize(im, (work_size, work_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    def load_mask(row):
        mp = row[mask_col] if (mask_col and pd.notna(row.get(mask_col))) \
            else os.path.join(mask_dir, stem(row[image_col]) + mask_ext)
        m = np.load(mp) if str(mp).endswith('.npy') else imread(mp)
        m = np.asarray(m); m = m[..., 0] if m.ndim == 3 else m
        return resize((m > 0).astype(float), (work_size, work_size), order=0, preserve_range=True).astype(np.uint8)

    imgs, msks, labs, paths = [], [], [], []
    for _, row in sub.iterrows():
        try:
            im = load_img(row[image_col])
        except Exception as e:
            print(f'[skip] {row[image_col]}: {e}'); continue
        try:
            mk = load_mask(row)
        except Exception:
            mk = np.zeros(im.shape, np.uint8)                      # CNN/VAE ignore it; ShapeEmbed skips zero masks
        try:
            lab = int(row[label_col])
        except Exception:
            continue
        imgs.append(im); msks.append(mk); labs.append(lab); paths.append(str(row[image_col]))
    n_mask = sum(int(m.sum() > 0) for m in msks)
    print(f"[train-fit] loaded {len(imgs)} {split} fish ({n_mask} with masks) for head fitting")
    return imgs, msks, np.array(labs, int), paths


def mask_to_contour_dm(mask, n_points, normalise='fro'):
    """Resample the largest contour to n_points -> (points[row,col], distance-matrix tensor [1,1,P,P]).
    Mirrors real_extractors._mask_to_distance_matrix so the DM matches what ShapeEmbed was trained on."""
    import torch
    from skimage.measure import find_contours
    cs = find_contours(mask.astype(float), 0.5)
    if not cs:
        return None, None
    c = max(cs, key=len)
    idx = np.linspace(0, len(c) - 1, n_points).astype(int)
    pts = c[idx]                                                    # (P, 2) as (row, col)
    dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).astype(np.float32)
    if normalise == 'fro':
        dm = dm / (np.linalg.norm(dm) + 1e-8)
    elif normalise == 'max':
        dm = dm / (dm.max() + 1e-8)
    return pts, torch.tensor(dm[None, None], dtype=torch.float32)


# --------------------------- rendering ---------------------------
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def top_mass(cam, frac=0.2):
    """Keep only the pixels carrying the top `frac` of total CAM mass; set the rest to NaN (transparent
    in imshow). A CONCENTRATED map -> small tight hot region; a DIFFUSE map -> large faint patch, so a
    weak map looks weak instead of a fake-confident normalized blob."""
    if not frac or frac >= 1:
        return cam
    flat = np.asarray(cam, float).ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    total = csum[-1] + 1e-12
    k = int(np.searchsorted(csum, frac * total)) + 1
    thr = flat[order][min(k, len(flat) - 1)]
    return np.where(np.asarray(cam) >= thr, cam, np.nan)


def save_heatmap(image, cam, path, title, mask=None, top_frac=0.2):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(_minmax(image), cmap='gray')
    ax.imshow(top_mass(cam, top_frac), cmap='jet', alpha=0.5, vmin=0, vmax=1)
    if mask is not None and np.asarray(mask).sum() > 0:                 # draw the larva outline
        ax.contour((np.asarray(mask) > 0).astype(float), levels=[0.5], colors='cyan', linewidths=0.7)
    ax.set_title(title, fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_contour(image, pts, sal, path, title, top_frac=0.2):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(_minmax(image), cmap='gray')
    ax.plot(pts[:, 1], pts[:, 0], color='0.6', lw=0.5, alpha=0.7)       # full outline (grey)
    thr = np.quantile(sal, 1 - top_frac) if top_frac else -np.inf       # colour only the top points
    hot = sal >= thr
    ax.scatter(pts[hot, 1], pts[hot, 0], c=sal[hot], cmap='jet', s=18, vmin=0, vmax=1)  # no colorbar: keep
    ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)      # panel size identical to heatmaps
    ax.set_title(title, fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------- metrics ---------------------------
def cam_metrics(cam, mask):
    """CAM localization vs the fish mask. A larva fills only a few % of the frame and layer4 CAMs are
    coarse, so raw energy-fraction ~ area even for a uniform map -- we therefore also report ENRICHMENT
    = (energy fraction in a region) / (that region's area fraction): >1 attends there MORE than chance,
    ~1 no preference, <1 avoids it."""
    tot = float(cam.sum()) + 1e-12
    fg = mask > 0
    npix = float(mask.size)
    area = float(fg.sum()) / npix + 1e-12
    regions = define_regions(mask)
    contain = float(cam[fg].sum()) / tot
    m = {'fish_area_frac': round(area, 4),
         'containment': contain,
         'enrichment': contain / area,
         'background_mass': float(cam[~fg].sum()) / tot}
    for r in ('body_ant', 'body_mid', 'body_post'):
        if r in regions and regions[r].any():
            ra = float(regions[r].sum()) / npix + 1e-12
            mass = float(cam[regions[r]].sum()) / tot
            m[f'mass_{r}'] = mass
            m[f'enr_{r}'] = mass / ra
    return m


def _prep_input(spec, image, path):
    """Build the input tensor: path-based prep (VAE full-res crop) or array-based prep (CNN)."""
    return spec['prep'](path) if spec.get('needs_path') else spec['prep'](image)


def extract_feature(spec, image, mask, path, device):
    """Re-extract the penultimate feature via the SAME forward pass Grad-CAM uses (no grad). Used to
    fit the matched head so its input distribution matches CAM time exactly."""
    import torch
    if spec['kind'] == 'image':
        x = _prep_input(spec, image, path).to(device)
        with torch.no_grad():
            f = spec['feature_fn'](spec['model'], x)
    else:                                                          # contour: distance-matrix -> z_mean
        _, dm = mask_to_contour_dm(mask, spec['matrix_size'], spec['normalise'])
        if dm is None:
            return None
        with torch.no_grad():
            f = spec['model'](dm.to(device))[3]
    return f.detach().cpu().numpy().ravel()


def _train_spearman(head, classes, X, y):
    """Sanity: does E[k] from the freshly-fit head track SC on the train features? (the Step-0 gate)."""
    import torch
    dev = next(head.parameters()).device
    with torch.no_grad():
        p = torch.softmax(head(torch.tensor(np.asarray(X, np.float32)).to(dev)), dim=1).cpu().numpy()
    Ek = p @ np.asarray(classes, float)
    return float(pd.Series(Ek).corr(pd.Series(np.asarray(y, float)), method='spearman'))


def spatial_corr(a, b, mask):
    sel = mask > 0
    x, y = a[sel].ravel(), b[sel].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------- backbone wiring ---------------------------
def build_backbones(methods, args, device):
    """Return {name: dict(kind, model, target_layer, feature_fn, prep)} for image methods, and for
    ShapeEmbed dict(kind='contour', model, matrix_size, normalise)."""
    bb = {}
    if 'CNN' in methods:
        m, cfg = load_cnn(args.cnn_weights, device, args.cnn_yaml, src=args.cnn_src)
        bb['CNN'] = dict(kind='image', model=m, target_layer=m.baseModel.layer4,
                         feature_fn=lambda mm, x: mm(x)['encoding'],
                         prep=lambda im: prep_cnn(im, cfg['size'], cfg['mean'], cfg['std']))
    if 'VAE' in methods:
        m = load_vae(args.vae_weights, device, latent_dim=args.vae_latent,
                     input_size=args.vae_size, src=args.vae_src)
        # VAE reads FULL-RES images and applies the CenterCrop(1536)->Resize(352) (frame='crop').
        # The CAM therefore lives in the 352 crop frame, not the shared 512 frame -> for now we compute
        # only E[k] here (the Spearman diagnostic); the CAM coordinate-mapping is Phase 2.
        bb['VAE'] = dict(kind='image', model=m, target_layer=m.vae.encoder.layer4, frame='crop',
                         needs_path=True, feature_fn=lambda mm, x: mm.vae.get_latent(x)[0],
                         prep=lambda pth: prep_vae_fullres(pth, args.vae_size, args.vae_crop))
    if 'ShapeEmbed' in methods:
        m, ms = load_shapeembed(args.shapeembed_weights, device, matrix_size=args.se_matrix,
                                padding=not args.se_no_padding, src=args.se_src)
        bb['ShapeEmbed'] = dict(kind='contour', model=m, matrix_size=ms,
                                normalise=None if args.se_normalise == 'none' else args.se_normalise)
    return bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--mask-dir'); ap.add_argument('--mask-col'); ap.add_argument('--mask-ext', default='.npy')
    ap.add_argument('--split', default='test'); ap.add_argument('--label-col', default='severity_score_adjusted')
    ap.add_argument('--n-per-class', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--work-size', type=int, default=512)
    ap.add_argument('--subset-file', default='../occlusion/results_occlusion/occlusion_subset_ids.txt',
                    help='reuse the SAME frozen subset as occlusion (default points at it)')
    ap.add_argument('--n-vis', type=int, default=3, help='overlays saved per SC class (metrics use all)')
    ap.add_argument('--methods', nargs='+', default=['CNN', 'VAE', 'ShapeEmbed'],
                    choices=['CNN', 'VAE', 'ShapeEmbed'])
    ap.add_argument('--cnn-weights'); ap.add_argument('--vae-weights'); ap.add_argument('--shapeembed-weights')
    ap.add_argument('--cnn-yaml'); ap.add_argument('--cnn-src'); ap.add_argument('--vae-src'); ap.add_argument('--se-src')
    ap.add_argument('--vae-latent', type=int); ap.add_argument('--vae-size', type=int, default=352)
    ap.add_argument('--vae-crop', type=int, default=1536,
                    help="VAE full-res CenterCrop size before resize (the inference transform)")
    ap.add_argument('--se-matrix', type=int)
    ap.add_argument('--se-normalise', choices=['none', 'fro', 'max'], default='fro')
    ap.add_argument('--se-no-padding', action='store_true')
    ap.add_argument('--train-split', default='training', help='manifest split used to fit the heads')
    ap.add_argument('--train-n-per-class', type=int, default=80)
    ap.add_argument('--train-mask-dir', help='masks for the TRAIN split (needed only for the ShapeEmbed '
                                             'head fit); default = --mask-dir')
    ap.add_argument('--probe-train', nargs='+', metavar='NAME=CSV',
                    help='FALLBACK archived feature CSV per method, used only if train re-extraction '
                         'yields too few fish (e.g. no train masks for ShapeEmbed)')
    ap.add_argument('--probe-label', default='label')
    ap.add_argument('--out', default='results_gradcam')
    a = ap.parse_args()

    device = _device(); print(f"[device] {device}")
    os.makedirs(a.out, exist_ok=True)
    ov = os.path.join(a.out, 'overlays'); os.makedirs(ov, exist_ok=True)
    cams_dir = os.path.join(a.out, 'cams'); os.makedirs(cams_dir, exist_ok=True)

    imgs, msks, meta = load_data_real(a.csv, label_col=a.label_col, split=a.split,
                                      mask_dir=a.mask_dir, mask_col=a.mask_col, mask_ext=a.mask_ext,
                                      n_per_class=a.n_per_class, work_size=a.work_size,
                                      seed=a.seed, subset_file=a.subset_file)
    if not imgs:
        raise SystemExit("[ERR] no fish loaded -- check --csv / --mask-dir / --mask-ext / --subset-file")
    labels = meta[a.label_col].astype(int).values if a.label_col in meta.columns else np.zeros(len(imgs), int)
    fish_ids = meta['fish_id'].values if 'fish_id' in meta.columns else np.arange(len(imgs)).astype(str)

    bb = build_backbones(a.methods, a, device)

    # ---- matched severity heads: fit on features RE-EXTRACTED by the SAME forward pass Grad-CAM uses,
    #      so the head sees the identical feature distribution (immune to preprocessing drift vs the
    #      archived features_*.csv). CNN/VAE ignore the mask; ShapeEmbed needs the train masks. ----
    trains = dict(kv.split('=', 1) for kv in a.probe_train) if a.probe_train else {}
    timgs, tmsks, tlab, tpaths = load_split_for_fit(
        a.csv, a.label_col, a.train_split, 'set', a.train_mask_dir or a.mask_dir, a.mask_col,
        a.mask_ext, a.train_n_per_class, a.work_size, a.seed)
    heads = {}
    for name in a.methods:
        spec = bb[name]
        X, y = [], []
        for img, msk, lab, pth in zip(timgs, tmsks, tlab, tpaths):
            if spec['kind'] == 'contour' and int(np.asarray(msk).sum()) == 0:
                continue                                           # ShapeEmbed needs a real train mask
            f = extract_feature(spec, img, msk, pth, device)
            if f is not None and np.isfinite(f).all():
                X.append(f); y.append(int(lab))
        if len(X) >= 20 and len(set(y)) >= 2:
            head, classes = fit_head_from_array(np.array(X), np.array(y), device)
            rho = _train_spearman(head, classes, X, y)
            flag = 'OK' if rho >= 0.3 else 'LOW -> extraction may lose the severity signal'
            print(f"[head:{name}] re-extracted {len(X)} train features  "
                  f"train Spearman(E[k],SC)={rho:.2f}  {flag}")
            heads[name] = (head, classes)
        elif name in trains:
            print(f"[head:{name}] only {len(X)} usable train features -> FALLBACK to archived CSV "
                  f"(warning: may be on the wrong feature scale)")
            heads[name] = fit_torch_head(trains[name], a.probe_label, device)
        else:
            raise SystemExit(f"[ERR] {name}: no usable train features (ShapeEmbed needs --train-mask-dir) "
                             f"and no --probe-train fallback CSV")

    paths = meta['image_path'].values if 'image_path' in meta.columns else np.array([''] * len(imgs))
    # per-class overlay budget
    saved = {c: 0 for c in np.unique(labels)}
    rows = []
    image_cams = {}                                                  # (fish_idx, method) -> cam for cross-corr
    for i, (img, msk) in enumerate(zip(imgs, msks)):
        y = int(labels[i]); fid = str(fish_ids[i]); pth = str(paths[i])
        do_overlay = saved.get(y, 0) < a.n_vis
        for name in a.methods:
            head, classes = heads[name]
            spec = bb[name]
            if spec['kind'] == 'image':
                if spec.get('frame') == 'crop':                    # VAE: CAM on the 352 crop...
                    cam_crop, ek = image_gradcam(spec['model'], spec['target_layer'], spec['feature_fn'],
                                                 head, classes, _prep_input(spec, img, pth),
                                                 out_hw=(a.vae_size, a.vae_size), device=device)
                    try:                                           # ...mapped back into the shared full frame
                        from skimage.io import imread as _imread
                        cam = place_crop_in_full(cam_crop, _imread(pth).shape[:2], img.shape, a.vae_crop)
                    except Exception:
                        cam = np.zeros(img.shape, np.float32)
                else:                                              # CNN: full-frame CAM
                    cam, ek = image_gradcam(spec['model'], spec['target_layer'], spec['feature_fn'],
                                            head, classes, _prep_input(spec, img, pth),
                                            out_hw=img.shape, device=device)
                image_cams[(i, name)] = cam                        # both in the 512 frame -> comparable
                m = cam_metrics(cam, msk); m.update(fish=fid, SC=y, method=name, Ek=round(ek, 3))
                rows.append(m)
                if do_overlay:
                    np.save(os.path.join(cams_dir, f'{name}_{fid}.npy'), cam.astype(np.float32))
                    save_heatmap(img, cam, os.path.join(ov, f'{name}_SC{y}_{fid}.png'),
                                 f'{name}  SC{y}  E[k]={ek:.2f}', mask=msk)
            else:  # contour
                pts, dm = mask_to_contour_dm(msk, spec['matrix_size'], spec['normalise'])
                if pts is None:
                    rows.append(dict(fish=fid, SC=y, method=name, Ek=float('nan')))
                    continue
                sal, ek = contour_saliency(spec['model'], dm, head, classes, device=device)
                rows.append(dict(fish=fid, SC=y, method=name, Ek=round(ek, 3),
                                 sal_ant=float(sal[:len(sal)//3].mean()),
                                 sal_mid=float(sal[len(sal)//3:2*len(sal)//3].mean()),
                                 sal_post=float(sal[2*len(sal)//3:].mean())))
                if do_overlay:
                    save_contour(img, pts, sal, os.path.join(ov, f'ShapeEmbed_SC{y}_{fid}.png'),
                                 f'ShapeEmbed  SC{y}  E[k]={ek:.2f}')
        if do_overlay:
            saved[y] = saved.get(y, 0) + 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'gradcam_perfish.csv'), index=False)

    # cross-backbone agreement (CNN vs VAE) where both exist
    if 'CNN' in a.methods and 'VAE' in a.methods:
        corr = []
        for i, msk in enumerate(msks):
            if (i, 'CNN') in image_cams and (i, 'VAE') in image_cams:
                corr.append(dict(fish=str(fish_ids[i]), SC=int(labels[i]),
                                 corr_cnn_vae=spatial_corr(image_cams[(i, 'CNN')], image_cams[(i, 'VAE')], msk)))
        if corr:
            cdf = pd.DataFrame(corr); cdf.to_csv(os.path.join(a.out, 'gradcam_cnn_vae_agreement.csv'), index=False)
            print(f"[agreement] CNN-vs-VAE spatial corr: mean={cdf['corr_cnn_vae'].mean():.3f} "
                  f"median={cdf['corr_cnn_vae'].median():.3f} (n={len(cdf)})")

    # summary: mean containment / ENRICHMENT / region mass per image method
    cols = [c for c in ['fish_area_frac', 'containment', 'enrichment', 'background_mass',
                        'enr_body_ant', 'enr_body_mid', 'enr_body_post'] if c in df.columns]
    summ = (df[df['method'].isin([m for m in a.methods if m != 'ShapeEmbed'])]
            .groupby('method')[cols].mean().round(3)) if (len(df) and cols) else pd.DataFrame()
    if len(summ):
        summ.to_csv(os.path.join(a.out, 'gradcam_summary.csv'))
        print('\n[summary] mean CAM containment / enrichment (=containment/area, >1 attends to fish) '
              '/ body-third enrichment:\n' + summ.to_string())
    print(f"\n[OK] -> {a.out}/  (overlays/, cams/, gradcam_perfish.csv"
          + (", gradcam_summary.csv, gradcam_cnn_vae_agreement.csv)" if len(summ) else ")"))


if __name__ == '__main__':
    main()
