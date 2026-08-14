#!/usr/bin/env python3
"""Grad-CAM for the image backbones (CNN, VAE encoder) and a contour-saliency variant for ShapeEmbed.

The target for each backbone is a linear severity head fit on its train representation (a scaler +
logistic regression folded into one nn.Linear), read out as the expected severity
E[k] = sum_k k * softmax(logits)_k. Back-propagating this scalar into the last convolutional block
gives the activation map. The driver (run_gradcam.py) handles data loading, preprocessing and IO.

Reference: Selvaraju et al., Grad-CAM, ICCV 2017.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- matched differentiable head ---------------------------
def _fold_logreg_into_linear(X, y, device='cpu'):
    """Fit StandardScaler+LogisticRegression on (X, y) and FOLD both into ONE differentiable
    nn.Linear(feat -> C). Returns (head, classes).

    Folding: logits = coef @ (x-mean)/scale + intercept
                    = (coef/scale) @ x + (intercept - coef @ (mean/scale))
    so the head takes RAW features and reproduces scaler+LogReg exactly, while staying differentiable
    (what Grad-CAM backprops). Matches standard_eval / the occlusion probe."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(X), y)
    W = clf.coef_ / sc.scale_                                       # (C, F)
    b = clf.intercept_ - (clf.coef_ * (sc.mean_ / sc.scale_)).sum(1)  # (C,)
    if W.shape[0] == 1:                                             # binary -> 2 logits for softmax/E[k]
        W = np.vstack([-W[0], W[0]]); b = np.array([-b[0], b[0]])
    classes = clf.classes_.astype(float)
    head = nn.Linear(X.shape[1], W.shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.tensor(W, dtype=torch.float32))
        head.bias.copy_(torch.tensor(b, dtype=torch.float32))
    head.to(device).eval()
    return head, classes


def fit_head_from_array(X, y, device='cpu'):
    """Fit the matched severity head on features RE-EXTRACTED by the Grad-CAM forward pass (X, y).
    This is the CORRECT path: the head sees the same feature distribution it will see at CAM time,
    so it is immune to preprocessing differences vs the archived features_*.csv."""
    head, classes = _fold_logreg_into_linear(X, y, device)
    print(f"[head] fit on {len(y)} re-extracted features: feat={np.shape(X)[1]} classes={classes.tolist()}")
    return head, classes


def fit_torch_head(train_csv, label_col='label', device='cpu'):
    """FALLBACK: fit the head on an archived feature CSV (fish_id, f0..fN, label). Only valid if that
    CSV was produced by the SAME extraction as the Grad-CAM forward pass -- otherwise the head is on
    the wrong feature scale (flat E[k]). Prefer fit_head_from_array on re-extracted features."""
    import os
    import pandas as pd
    df = pd.read_csv(train_csv)
    y = df[label_col].astype(int).values
    X = df.drop(columns=[c for c in ('fish_id', label_col) if c in df.columns]).select_dtypes('number').values.astype(float)
    head, classes = _fold_logreg_into_linear(X, y, device)
    print(f"[head] {os.path.basename(train_csv)}: feat={X.shape[1]} classes={classes.tolist()} "
          f"(folded scaler+LogReg, from ARCHIVED csv)")
    return head, classes


def expected_severity(head, feat, classes):
    """E[k] = sum_k k * softmax(head(feat))_k -- the differentiable severity scalar Grad-CAM targets."""
    ks = torch.as_tensor(classes, dtype=torch.float32, device=feat.device)
    return (torch.softmax(head(feat), dim=1) * ks).sum()


# --------------------------- Grad-CAM (image backbones) ---------------------------
class GradCAM:
    """Hooks a conv layer; after a backward pass, compute(out_hw) returns the normalized, upsampled CAM."""
    def __init__(self, target_layer):
        self._A = None
        self._dA = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self._A = out

    def _save_grad(self, module, grad_in, grad_out):
        self._dA = grad_out[0]

    def remove(self):
        self._h1.remove(); self._h2.remove()

    def compute(self, out_hw):
        if self._A is None or self._dA is None:
            raise RuntimeError("GradCAM: no activation/gradient captured -- did backward() run and did "
                               "the forward pass go through the hooked layer?")
        A, dA = self._A.detach(), self._dA.detach()                # [1, C, h, w]
        alpha = dA.mean(dim=(2, 3), keepdim=True)                  # channel weights = GAP of grads
        cam = F.relu((alpha * A).sum(dim=1, keepdim=True))         # [1, 1, h, w]
        cam = F.interpolate(cam, size=out_hw, mode='bilinear', align_corners=False)[0, 0]
        cam = cam - cam.min()
        mx = float(cam.max())
        return (cam / mx).cpu().numpy() if mx > 0 else cam.cpu().numpy()


def image_gradcam(backbone, target_layer, feature_fn, head, classes, x, out_hw, device='cpu'):
    """Grad-CAM for one image. feature_fn(backbone, x) -> penultimate feature tensor [1, F]; the target
    is E[k] through the matched head. Returns (cam2d in [0,1] at out_hw, E[k] value)."""
    backbone.eval()
    x = x.to(device).requires_grad_(True)                          # force grad through a frozen backbone
    cam = GradCAM(target_layer)
    try:
        backbone.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        feat = feature_fn(backbone, x)                             # [1, F], graph intact
        Ek = expected_severity(head, feat, classes)
        Ek.backward()
        out = cam.compute(out_hw)
    finally:
        cam.remove()
    return out, float(Ek.detach())


# --------------------------- contour saliency (ShapeEmbed) ---------------------------
def contour_saliency(se_model, dm, head, classes, device='cpu'):
    """Saliency per CONTOUR POINT for the shape backbone: backprop E[k] to the input distance matrix
    and reduce |grad| over each point's row+column. dm: [1,1,P,P] tensor. Returns (saliency[P] in
    [0,1], E[k]).  se_model(dm) -> (preproc, recon, z, z_mean, z_log_var, scale); we use z_mean."""
    se_model.eval()
    dm = dm.to(device).clone().requires_grad_(True)
    out = se_model(dm)
    z_mean = out[3]
    se_model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    Ek = expected_severity(head, z_mean, classes)
    Ek.backward()
    g = dm.grad.abs()[0, 0]                                         # [P, P]
    sal = g.sum(0) + g.sum(1)                                       # contribution of each point
    mx = float(sal.max())
    sal = (sal / mx) if mx > 0 else sal
    return sal.detach().cpu().numpy(), float(Ek.detach())
