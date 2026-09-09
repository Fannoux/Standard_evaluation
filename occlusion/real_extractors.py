#!/usr/bin/env python3
"""Feature extractors and read-outs for the occlusion analysis.

Each extractor maps (image, mask) to a 1-D feature vector; predictors map features to a scalar.
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
    """CNN extractor: min-max, resize, 1->3 ch, normalize; returns the penultimate encoding."""
    import torch
    mean = IMAGENET_MEAN if mean is None else np.asarray(mean, float)
    std = IMAGENET_STD if std is None else np.asarray(std, float)
    model.eval().to(device)
    def f(image, mask):
        x = _resize(_minmax(image), size)
        x = np.repeat(x[None], 3, axis=0)                       # 1 -> 3 channels
        x = (x - mean[:, None, None]) / std[:, None, None]
        t = torch.tensor(x[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(t)
        return out['encoding'].cpu().numpy().ravel()
    return f


# =============================== VAE ===============================
def extract_vae(model, device='cpu', size=352):
    """VAE extractor: resize, per-image standardize; returns the posterior mean mu."""
    import torch
    model.eval().to(device)
    def f(image, mask):
        x = _resize(image.astype(np.float32), size)
        x = (x - x.mean()) / (x.std() + 1e-8)
        t = torch.tensor(x[None, None], dtype=torch.float32, device=device)   # [1,1,H,W]
        with torch.no_grad():
            mu, _ = model.vae.get_latent(t)
        return mu.cpu().numpy().ravel()
    return f


# =============================== RegionProps ===============================
def extract_regionprops():
    """19 classical descriptors from the mask (skimage). Order fixed for reproducibility."""
    from skimage.measure import regionprops, label
    def f(image, mask):
        lab = label(mask > 0)
        props = regionprops(lab)
        if not props:
            return np.zeros(19, np.float32)
        p = max(props, key=lambda r: r.area)                   # largest component = the sample
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
    idx = np.linspace(0, len(c) - 1, n_points).astype(int)
    pts = c[idx]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return d.astype(np.float32)

def extract_shapeembed(model, device='cpu', n_points=256, normalise='fro'):
    """ShapeEmbed extractor: mask -> distance matrix -> posterior mean z_mean (n_points must match matrix_size)."""
    import torch
    model.eval().to(device)
    def f(image, mask):
        dm = _mask_to_distance_matrix(mask, n_points)
        if normalise == 'fro':
            dm = dm / (np.linalg.norm(dm) + 1e-8)              # Frobenius norm
        elif normalise == 'max':
            dm = dm / (dm.max() + 1e-8)
        t = torch.tensor(dm[None, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(t)                                     # tuple; z_mean is index 3
        return out[3].cpu().numpy().ravel()
    return f


# =============================== predictor (features -> label) ===============================
def make_predictor_probe(scaler, clf):
    """Predictor from a fitted scaler + classifier: E[k] = sum_k k P(k)."""
    ks = clf.classes_.astype(float)
    def predict(feat):
        z = scaler.transform(feat.reshape(1, -1))
        P = clf.predict_proba(z)[0]
        return float(P @ ks)
    return predict

def make_predictor_latent():
    """Probe-free predictor: latent magnitude ||features||."""
    return lambda feat: float(np.linalg.norm(feat))
