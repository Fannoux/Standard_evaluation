#!/usr/bin/env python3
"""
CLI driver for the occlusion interpretability analysis.

Loads a stratified test subset once, runs every requested method on the same fish, and writes a
combined results CSV plus a per-method region table. RegionProps needs no model weights, so
`--methods RegionProps` validates the pipeline before wiring the neural encoders.

Example (see run_occlusion.slurm):
  python run_occlusion.py --csv manifest.csv --mask-dir masks --mask-ext .npy \
      --split test --n-per-class 40 --methods RegionProps CNN VAE ShapeEmbed \
      --cnn-weights cnn.pth --vae-weights vae.pth --shapeembed-weights se.pth \
      --predict latent --out results_occlusion
"""
import argparse, os
import pandas as pd
from occlusion_interpretability import occlusion_sensitivity, load_data_real
from real_extractors import (extract_cnn, extract_vae, extract_regionprops,
                                      extract_shapeembed, make_predictor_latent, make_predictor_probe)


def _device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


# ---------------- model-load hooks (wired to the sibling repos) ----------------
# The model classes live in the sibling Ziram repos; add them to sys.path so they import.
_HERE = os.path.dirname(os.path.abspath(__file__))
CNN_SRC = os.path.join(_HERE, '..', '..', 'scripts_cnn')       # pyimagesearch.classifier / .config
VAE_SRC = os.path.join(_HERE, '..', '..', 'VAE')        # src_vae.model.VAEModel
SE_SRC  = os.path.join(_HERE, '..', '..', 'ShapeEmbedLite')    # utils.models.MyNet


def _add_path(p):
    import sys
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_cnn(weights, device, yaml=None, src=None):
    """Larval_MLClassifier, loaded exactly as evaluation.py (model_kwargs come from run_info.yaml).
    Returns (model, cfg) where cfg carries the run's im_size/mean/std for matched preprocessing.

    NOTE: the classifier builds its backbone via torch.hub.load('pytorch/vision:v0.10.0','resnet18'),
    whose OLD torchvision source does `from torch.onnx.symbolic_opset9 import _cast_Long` -- removed in
    modern torch -> ImportError. We don't need the hub download: build an identical resnet18 from the
    INSTALLED torchvision (the trained state_dict overwrites the weights anyway). Shim is scoped to
    resnet18 and restored right after construction."""
    import torch, torchvision
    _add_path(src or CNN_SRC)
    from pyimagesearch.classifier import Larval_MLClassifier
    from pyimagesearch.config import params_fromYAML
    import torch.hub as _hub
    _orig_load = _hub.load
    def _load_shim(*a, **k):
        name = k.get('model') if 'model' in k else (a[1] if len(a) > 1 else None)
        if name == 'resnet18':
            return torchvision.models.resnet18(weights=None)
        return _orig_load(*a, **k)
    _hub.load = _load_shim
    try:
        if yaml is None:
            yaml = os.path.join(os.path.dirname(os.path.abspath(weights)), 'run_info.yaml')
        data_kwargs, model_kwargs, *_ = params_fromYAML(yaml)
        model = Larval_MLClassifier(**model_kwargs).to(device)
    finally:
        _hub.load = _orig_load
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    cfg = dict(size=data_kwargs['im_size'], mean=data_kwargs['mean'], std=data_kwargs['std'])
    return model, cfg


def load_vae(weights, device, latent_dim=None, capacity=16, depth=4, input_size=352, src=None):
    """the VAE. Occlusion only needs the ENCODER (image -> posterior mean mu).

    Two checkpoint flavours are handled, auto-detected from the keys:
      * ResNet-18 encoder VAE (the current model): keys 'vae.encoder.conv1', 'vae.encoder.layerN...',
        'vae.mean' = Linear(512, L). We REBUILD the encoder from the checkpoint (torchvision resnet18
        with a matched first conv + a Linear head), so it does NOT depend on the exact repo version.
      * plain-conv VAEModel (older repo code): keys 'vae.encoder.0.0...', 'vae.mean' = Linear(HUGE, L).
        Built from src_vae.model.VAEModel at --vae-src.
    In both cases the returned object exposes model.vae.get_latent(x) -> (mean, logvar)."""
    import torch
    sd = torch.load(weights, map_location=device)

    if 'vae.encoder.conv1.weight' in sd:                    # ---- ResNet-18 encoder VAE ----
        import torch.nn as nn, torchvision
        in_ch = int(sd['vae.encoder.conv1.weight'].shape[1])
        lat = latent_dim or int(sd['vae.mean.weight'].shape[0])
        feat = int(sd['vae.mean.weight'].shape[1])          # resnet feature dim (512 for resnet18)

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                net = torchvision.models.resnet18(weights=None)
                net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                net.fc = nn.Identity()                      # expose the 512-d avgpool feature
                self.encoder = net
                self.mean = nn.Linear(feat, lat)
                self.var = nn.Linear(feat, lat)
            def get_latent(self, x):
                h = self.encoder(x)
                return self.mean(h), self.var(h)

        class _Wrap(nn.Module):
            def __init__(self): super().__init__(); self.vae = _Enc()

        model = _Wrap().to(device)
        info = model.load_state_dict(sd, strict=False)      # decoder keys are unexpected -> ignored
        need = [k for k in info.missing_keys if k.startswith(('vae.encoder', 'vae.mean', 'vae.var'))]
        if need:
            raise SystemExit(f"[ERR] VAE encoder weights did not fully load; missing e.g. {need[:3]}")
        model.eval()
        print(f"[vae] rebuilt ResNet18 encoder from checkpoint (in_ch={in_ch}, feat={feat}, latent={lat}); "
              f"ignored {len(info.unexpected_keys)} decoder key(s)")
        return model

    # ---- plain-conv VAEModel from the repo ----
    _add_path(src or VAE_SRC)
    from src_vae.model import VAEModel
    if latent_dim is None:
        latent_dim = int(sd['vae.mean.weight'].shape[0])
    model = VAEModel(input_dim=(1, input_size, input_size), latent_dim=latent_dim,
                     capacity=capacity, depth=depth, device=str(device))
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


def load_shapeembed(weights, device, matrix_size=None, latent_dim=None,
                    space_dim=2, padding=True, decoder_depth=5, src=None):
    """ShapeEmbedLite MyNet. latent_dim and matrix_size are auto-inferred from the checkpoint:
    z_mean.weight -> latent_dim; decoder_layers.0.weight -> matrix_size (= hidden[0] / 2**(depth-1)).
    Returns (model, matrix_size) so the extractor builds distance matrices of the right size."""
    import torch
    _add_path(src or SE_SRC)
    from utils.models import MyNet
    sd = torch.load(weights, map_location=device)
    if latent_dim is None:
        latent_dim = int(sd['z_mean.weight'].shape[0])
    if matrix_size is None:
        # DecoderMLP's final Linear outputs space_dim * num_points, and num_points == matrix_size.
        import re
        li = max(int(re.match(r'decoder\.decoder_layers\.(\d+)\.weight', k).group(1))
                 for k in sd if re.match(r'decoder\.decoder_layers\.\d+\.weight', k))
        matrix_size = int(sd[f'decoder.decoder_layers.{li}.weight'].shape[0]) // space_dim
    dhl = [2 ** (i - 1) * matrix_size for i in range(decoder_depth, 0, -1)]
    model = MyNet(latent_dim=latent_dim, matrix_size=matrix_size, space_dim=space_dim,
                  padding=padding, decoder_hidden_layers=dhl)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, matrix_size


def build_extractors(methods, args, device):
    ext = {}
    if 'RegionProps' in methods:
        ext['RegionProps'] = extract_regionprops()                      # no model needed
    if 'CNN' in methods:
        m, cfg = load_cnn(args.cnn_weights, device, args.cnn_yaml, src=args.cnn_src)
        ext['CNN'] = extract_cnn(m, device, size=cfg['size'], mean=cfg['mean'], std=cfg['std'])
    if 'VAE' in methods:
        m = load_vae(args.vae_weights, device, latent_dim=args.vae_latent,
                     input_size=args.vae_size, src=args.vae_src)
        ext['VAE'] = extract_vae(m, device, size=args.vae_size)
    if 'ShapeEmbed' in methods:
        m, ms = load_shapeembed(args.shapeembed_weights, device, matrix_size=args.se_matrix,
                                padding=not args.se_no_padding, src=args.se_src)
        norm = None if args.se_normalise == 'none' else args.se_normalise
        ext['ShapeEmbed'] = extract_shapeembed(m, device, n_points=ms, normalise=norm)
    return ext


def fit_probe(train_csv, label_col='label'):
    """Fit the severity read-out on a method's TRAIN feature CSV -- reproduces standard_eval's
    report_holdout exactly: StandardScaler.fit(train) then LogisticRegression on the SCALED train.
    Returns (scaler, clf); make_predictor_probe turns it into E[k]=sum_k k*P(k). No .pkl needed.
    NB: the CSV feature order MUST match what the occlusion extractor emits (guaranteed for the learned
    latents CNN/VAE/ShapeEmbed; for RegionProps use a CSV built by THIS repo's extractor)."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    df = pd.read_csv(train_csv)
    y = df[label_col].astype(int).values
    X = df.drop(columns=[c for c in ('fish_id', label_col) if c in df.columns]).select_dtypes('number').values.astype(float)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000).fit(scaler.transform(X), y)
    print(f"[probe] fit on {os.path.basename(train_csv)}: n={len(y)} dims={X.shape[1]} "
          f"classes={np.bincount(y).tolist()}")
    return scaler, clf


def build_pipelines(methods, args, device):
    """Return {name: (extract_fn, predict_fn)}. In 'latent' mode predict is the shared ||features||;
    in 'probe' mode each method gets its OWN read-out fit from its --probe-train NAME=path CSV."""
    exts = build_extractors(methods, args, device)
    if args.predict != 'probe':
        predict = make_predictor_latent()
        return {name: (ext, predict) for name, ext in exts.items()}
    trains = dict(kv.split('=', 1) for kv in (args.probe_train or []))
    missing = [m for m in exts if m not in trains]
    if missing:
        raise SystemExit(f"[ERR] --predict probe needs --probe-train NAME=path for each method; "
                         f"missing: {missing}")
    pipe = {}
    for name, ext in exts.items():
        scaler, clf = fit_probe(trains[name], args.probe_label)
        pipe[name] = (ext, make_predictor_probe(scaler, clf))
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='manifest CSV (image_path, set, severity_score_adjusted)')
    ap.add_argument('--mask-dir'); ap.add_argument('--mask-col'); ap.add_argument('--mask-ext', default='.npy')
    ap.add_argument('--split', default='test'); ap.add_argument('--label-col', default='severity_score_adjusted')
    ap.add_argument('--n-per-class', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--work-size', type=int, default=512)
    ap.add_argument('--subset-file', default='occlusion_subset_ids.txt', help='freeze/reuse the exact fish set')
    ap.add_argument('--methods', nargs='+', default=['RegionProps'],
                    choices=['RegionProps', 'CNN', 'VAE', 'ShapeEmbed'])
    ap.add_argument('--cnn-weights'); ap.add_argument('--vae-weights'); ap.add_argument('--shapeembed-weights')
    ap.add_argument('--cnn-yaml', help='run_info.yaml for the CNN (default: next to --cnn-weights)')
    ap.add_argument('--cnn-src', help='path to the scripts_cnn repo (default: ../../scripts_cnn)')
    ap.add_argument('--vae-src', help="path to the VAE repo, i.e. the dir CONTAINING src_vae/ "
                                      "(default: ../../VAE)")
    ap.add_argument('--se-src', help='path to the ShapeEmbedLite repo (default: ../../ShapeEmbedLite)')
    ap.add_argument('--vae-latent', type=int, help='VAE latent dim (default: infer from checkpoint)')
    ap.add_argument('--vae-size', type=int, default=352, help='VAE input size (matches training resize)')
    ap.add_argument('--se-matrix', type=int, help='ShapeEmbed matrix/contour size (default: infer)')
    ap.add_argument('--se-normalise', choices=['none', 'fro', 'max'], default='fro',
                    help="distance-matrix normalisation; match the trained run (normfro -> 'fro')")
    ap.add_argument('--se-no-padding', action='store_true',
                    help='disable ShapeEmbed circular padding (training used cir_pad -> default ON)')
    ap.add_argument('--predict', choices=['latent', 'probe'], default='latent')
    ap.add_argument('--probe-train', nargs='+', metavar='NAME=CSV',
                    help='probe mode: per-method TRAIN feature CSV, e.g. CNN=features_train.csv '
                         'RegionProps=rp_train.csv (fit on the fly, no .pkl needed)')
    ap.add_argument('--probe-label', default='label', help="label column in the train CSVs (shared format)")
    ap.add_argument('--out', default='results_occlusion')
    a = ap.parse_args()

    device = _device(); print(f"[device] {device}")
    os.makedirs(a.out, exist_ok=True)             # make OUT first so the subset-freeze can write into it
    imgs, msks, meta = load_data_real(a.csv, label_col=a.label_col, split=a.split,
                                      mask_dir=a.mask_dir, mask_col=a.mask_col, mask_ext=a.mask_ext,
                                      n_per_class=a.n_per_class, work_size=a.work_size,
                                      seed=a.seed, subset_file=a.subset_file)
    if not imgs:
        raise SystemExit("[ERR] no fish loaded -- check --csv / --mask-dir / --mask-ext")

    pipe = build_pipelines(a.methods, a, device)
    summary, per_fish = [], []
    for name, (extract, predict) in pipe.items():
        df, summ = occlusion_sensitivity(imgs, msks, extract, predict)
        df['method'] = name; per_fish.append(df)
        summ.to_frame('mean_delta').to_csv(os.path.join(a.out, f'occlusion_regions_{name}.csv'))
        bg, fg = summ.get('background', 0.0), summ.get('foreground', 0.0)
        frac = bg / (bg + fg) if (bg + fg) else float('nan')
        row = {'method': name, 'n_fish': len(imgs), 'background_fraction': round(frac, 3)}
        row.update({r: round(float(v), 4) for r, v in summ.items()})
        summary.append(row)
        print(f"[{name}] background_fraction={frac:.3f}  " +
              "  ".join(f"{r}={v:.3f}" for r, v in summ.items()))

    pd.concat(per_fish).to_csv(os.path.join(a.out, 'occlusion_perfish_all.csv'), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(a.out, 'occlusion_summary.csv'), index=False)
    print(f"\n[OK] -> {a.out}/occlusion_summary.csv (+ per-method region tables, per-fish CSV)")


if __name__ == '__main__':
    main()
