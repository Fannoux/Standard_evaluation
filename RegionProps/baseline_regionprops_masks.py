#!/usr/bin/env python3
"""
RegionProps baseline on binary segmentation masks (the ShapeEmbed comparison baseline).

Extracts the 19 scikit-image shape descriptors used by the ShapeEmbed paper, directly
from the masks (no thresholding needed — the mask IS the segmentation). For each mask
it keeps the largest connected component (the fish) and computes:
  area, convex_area, perimeter, axis_major_length, axis_minor_length, extent,
  eccentricity, solidity, feret_diameter_max, hu_moments (7), bbox (w, h, aspect) = 19.

Usage:
    python baseline_regionprops_masks.py --masks maskF0 --output regionprops_maskF0.csv
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

FEATURES = ['area', 'convex_area', 'perimeter', 'axis_major_length', 'axis_minor_length',
            'extent', 'eccentricity', 'solidity', 'feret_diameter_max',
            'hu_moment_0', 'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
            'hu_moment_4', 'hu_moment_5', 'hu_moment_6',
            'bbox_width', 'bbox_height', 'bbox_aspect_ratio']


def features_from_mask(path):
    m = np.array(Image.open(path).convert('L'))
    binary = m > 0
    lab = label(binary)
    if lab.max() == 0:
        return None
    r = max(regionprops(lab), key=lambda x: x.area)   # largest component = the fish
    minr, minc, maxr, maxc = r.bbox
    h, w = maxr - minr, maxc - minc
    feats = {
        'area': float(r.area), 'convex_area': float(r.convex_area), 'perimeter': float(r.perimeter),
        'axis_major_length': float(r.axis_major_length), 'axis_minor_length': float(r.axis_minor_length),
        'extent': float(r.extent), 'eccentricity': float(r.eccentricity), 'solidity': float(r.solidity),
        'feret_diameter_max': float(r.feret_diameter_max),
        'bbox_width': float(w), 'bbox_height': float(h),
        'bbox_aspect_ratio': float(w / h) if h else 0.0,
    }
    for i, hu in enumerate(r.moments_hu):
        feats[f'hu_moment_{i}'] = float(hu)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', default='maskF0', help='folder of binary mask PNGs')
    ap.add_argument('--output', default='regionprops_maskF0.csv')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.masks, '*.png')))
    rows, errors = [], 0
    for i, f in enumerate(files):
        feats = features_from_mask(f)
        if feats is None:
            errors += 1
            continue
        stem = os.path.basename(f)[:-4]
        rows.append({'mask': os.path.basename(f), 'stem': stem,
                     'CO6': stem,  # mask stem == CO6 image basename, for label joining
                     **feats})
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(files)}]")

    df = pd.DataFrame(rows, columns=['mask', 'stem', 'CO6'] + FEATURES)
    df.to_csv(args.output, index=False)
    print(f"\n[OK] {len(df)} masks -> {args.output}  ({errors} empty/failed)")
    print(f"[INFO] feature columns: {len(FEATURES)} (the ShapeEmbed-paper region-props set)")


if __name__ == '__main__':
    main()
