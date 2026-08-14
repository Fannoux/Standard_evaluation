#!/usr/bin/env python3
"""Feature extractors and severity read-outs for the occlusion analysis.

Each extractor maps (image, mask) to a 1-D feature vector: CNN and VAE from the image, RegionProps
and ShapeEmbed from the mask. Predictors map features to a scalar severity, either through a fitted
scaler + classifier (E[k] = sum_k k P(k)) or the feature norm. Torch is imported lazily.
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================== preprocessing ===============================
def _minmax(im):
    im = im.astype(np.float32); lo, hi = im.min(), im.max()
    return (im - lo) / (hi - lo + 1e-8)

def _resize(im, size):
    from skimage.transform import resize
    return resize(im, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


# =============================== CNN ===============================
def extract_cnn(model, device='cpu', size=352, mean=None, std=None):
    """model = loaded Larval_MLClassifier; forward returns {'logits','encoding'} (512-d avgpool).
    Preprocessing mirrors ZiramDataset EXACTLY: per-image min-max -> resize to im_size -> 1->3 ch ->
    Normalize(mean,std). Pass the run's mean/std (from the YAML); defaults to ImageNet stats."""
    import torch
    mean = IMAGENET_MEAN if mean is None else np.asarray(mean, float)
    std = IMAGENET_STD if std is None else np.asarray(std, float)
    model.eval().to(device)
    def f(image, mask):
        x = _resize(_minmax(image), size)                       # per-image min-max, resize
        x = np.repeat(x[None], 3, axis=0)                       # 1 -> 3 channels
        x = (x - mean[:, None, None]) / std[:, None, None]      # Normalize(mean,std)
        t = torch.tensor(x[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(t)
        return out['encoding'].cpu().numpy().ravel()            # 512-d penultimate features
    return f


# =============================== VAE ===============================
def extract_vae(model, device='cpu', size=352):
    """model = loaded VAEModel. Encoding = posterior mean from model.vae.get_latent(x).
    Preprocessing mirrors data.py inference: resize to 352 -> per-image standardization -> 1 channel.
    (NB: training CenterCrops 1536 BEFORE the resize; here we resize the work-size crop directly.
    Occlusion measures deltas through the same pipeline, but keep this FOV difference in mind if you
    ever compare absolute mu against the saved encodings.)"""
    import torch
    model.eval().to(device)
    def f(image, mask):
        x = _resize(image.astype(np.float32), size)
        x = (x - x.mean()) / (x.std() + 1e-8)                  # A.Normalize(normalization='image')
        t = torch.tensor(x[None, None], dtype=torch.float32, device=device)   # [1,1,H,W]
        with torch.no_grad():
            mu, _ = model.vae.get_latent(t)                    # (mean, logvar) -> take mean
        return mu.cpu().numpy().ravel()
    return f


# =============================== RegionProps (runnable) ===============================
def extract_regionprops():
    """19 classical descriptors from the mask (skimage). Order fixed for reproducibility."""
    from skimage.measure import regionprops, label
    def f(image, mask):
        lab = label(mask > 0)
        props = regionprops(lab)
        if not props:
            return np.zeros(19, np.float32)
        p = max(props, key=lambda r: r.area)                   # largest component = the larva
        minr, minc, maxr, maxc = p.bbox
        bw, bh = (maxc - minc), (maxr - minr)
        feats = [p.area, p.perimeter, p.axis_major_length, p.axis_minor_length,
                 *p.moments_hu,                                # 7 Hu moments
                 bw, bh, bw / (bh + 1e-8),                     # bbox width, height, aspect
                 p.area_convex, p.extent, p.eccentricity, p.solidity, p.feret_diameter_max]
        return np.array(feats, np.float32)                     # length 19
    return f


# =============================== ShapeEmbed ===============================
def _mask_to_distance_matrix(mask, n_points=64):
    """Largest external contour -> resample to n_points -> pairwise Euclidean distance matrix."""
    from skimage.measure import find_contours
    cs = find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.zeros((n_points, n_points), np.float32)
    c = max(cs, key=len)                                       # longest contour
    idx = np.linspace(0, len(c) - 1, n_points).astype(int)     # TODO: match ShapeEmbedLite's resampling
    pts = c[idx]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return d.astype(np.float32)

def extract_shapeembed(model, device='cpu', n_points=256, normalise='fro'):
    """model = loaded MyNet (ShapeEmbedLite). forward(x) -> (preproc, recon, z, z_mean, z_log_var,
    scale); encoding = z_mean (index 3; model.eval() makes reparameterize return the mean).
    Feed a raw distance matrix; `normalise` must match how the TRAINING DMs were built
    (ConvertBinaryMasksToDMs -- the '...normfro' runs used Frobenius). n_points MUST equal the model's
    matrix_size (auto-detected in load_shapeembed and passed here)."""
    import torch
    model.eval().to(device)
    def f(image, mask):
        dm = _mask_to_distance_matrix(mask, n_points)
        if normalise == 'fro':
            dm = dm / (np.linalg.norm(dm) + 1e-8)              # Frobenius (matches the normfro runs)
        elif normalise == 'max':
            dm = dm / (dm.max() + 1e-8)
        t = torch.tensor(dm[None, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(t)                                     # tuple; z_mean is index 3
        return out[3].cpu().numpy().ravel()                    # posterior mean mu
    return f


# =============================== predictor (features -> severity) ===============================
def make_predictor_probe(scaler, clf):
    """(A) reuse your standard_eval scaler + fitted classifier -> E[k] = sum_k k P(k)."""
    ks = clf.classes_.astype(float)
    def predict(feat):
        z = scaler.transform(feat.reshape(1, -1))
        P = clf.predict_proba(z)[0]
        return float(P @ ks)
    return predict

def make_predictor_latent():
    """(B) probe-free: severity proxy = latent magnitude; occlusion then measures ||Δfeatures||."""
    return lambda feat: float(np.linalg.norm(feat))

# --------------------------------------------------------------------------------------
# WIRING EXAMPLE (pseudo — fill model loads + your test image/mask paths):
#
#   from occlusion_interpretability import occlusion_sensitivity, load_data_real
#   import joblib, torch
#   scaler, clf = joblib.load('cnn_scaler.pkl'), joblib.load('cnn_logreg.pkl')   # from standard_eval
#   model = load_cnn('cnn_best312.pth')
#   extract  = extract_cnn(model)
#   predict  = make_predictor_probe(scaler, clf)            # or make_predictor_latent()
#   imgs, msks = load_data_real(image_dir, mask_dir, fish_ids)   # both, aligned
#   df, summ = occlusion_sensitivity(imgs, msks, extract, predict)
# --------------------------------------------------------------------------------------
