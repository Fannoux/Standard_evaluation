"""
Baseline feature extraction using scikit-image region properties.

Extracts 19 region properties features from images as described in:
"We extract 19 region properties features that pertain to shape using the
region-props functionality of the scikit-image library"

Features extracted:
  1. area
  2. convex_area
  3. perimeter
  4. axis_major_length
  5. axis_minor_length
  6. extent
  7. eccentricity
  8. solidity
  9. feret_diameter_max
  10-16. hu_moments (7 features)
  17-19. bbox dimensions (width, height, aspect_ratio)

Usage:
    python baseline_region_props.py --metadata <csv_path> --output <output_csv> [--channel CO1]
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from skimage import io
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
import warnings

warnings.filterwarnings('ignore')


def extract_region_properties(image_path, channel='CO5'):
    """
    Extract 19 region properties from a single image.

    Parameters
    ----------
    image_path : str
        Path to image file
    channel : str
        Which channel to use ('CO1' for fluorescence, 'CO5' for brightfield)
        Used for thresholding strategy selection

    Returns
    -------
    features : dict
        Dictionary with 19 region properties
    """
    try:
        img = io.imread(image_path)

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        # Threshold using Otsu's method
        thresh = threshold_otsu(img)
        binary = img > thresh

        # Label connected components
        labeled_img = label(binary)

        # If no regions found, return NaN
        if labeled_img.max() == 0:
            return _get_nan_features()

        # Get the largest region (assuming the fish is the largest object)
        regions = regionprops(labeled_img)
        largest_region = max(regions, key=lambda r: r.area)

        # Extract features
        features = {
            'area': float(largest_region.area),
            'convex_area': float(largest_region.convex_area),
            'perimeter': float(largest_region.perimeter),
            'axis_major_length': float(largest_region.axis_major_length),
            'axis_minor_length': float(largest_region.axis_minor_length),
            'extent': float(largest_region.extent),
            'eccentricity': float(largest_region.eccentricity),
            'solidity': float(largest_region.solidity),
            'feret_diameter_max': float(largest_region.feret_diameter_max),
        }

        # Hu moments (7 features)
        hu_moments = largest_region.moments_hu
        for i, hu in enumerate(hu_moments):
            features[f'hu_moment_{i}'] = float(hu)

        # Bounding box dimensions
        minr, minc, maxr, maxc = largest_region.bbox
        bbox_height = maxr - minr
        bbox_width = maxc - minc
        bbox_aspect = bbox_width / bbox_height if bbox_height > 0 else 0

        features['bbox_width'] = float(bbox_width)
        features['bbox_height'] = float(bbox_height)
        features['bbox_aspect_ratio'] = float(bbox_aspect)

        return features

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return _get_nan_features()


def _get_nan_features():
    """Return NaN features for failed images."""
    features = {
        'area': np.nan,
        'convex_area': np.nan,
        'perimeter': np.nan,
        'axis_major_length': np.nan,
        'axis_minor_length': np.nan,
        'extent': np.nan,
        'eccentricity': np.nan,
        'solidity': np.nan,
        'feret_diameter_max': np.nan,
    }
    for i in range(7):
        features[f'hu_moment_{i}'] = np.nan
    features['bbox_width'] = np.nan
    features['bbox_height'] = np.nan
    features['bbox_aspect_ratio'] = np.nan

    return features


def process_dataset(metadata_csv, output_csv, channel='CO5'):
    """
    Process all samples in metadata CSV and extract features.

    Parameters
    ----------
    metadata_csv : str
        Path to metadata CSV (e.g., Gronske_F0_Data_TRAIN.csv)
    output_csv : str
        Output CSV path
    channel : str
        Which channel to extract ('CO1' or 'CO5')
    """
    feature_names = [
        'area', 'convex_area', 'perimeter', 'axis_major_length',
        'axis_minor_length', 'extent', 'eccentricity', 'solidity',
        'feret_diameter_max', 'hu_moment_0', 'hu_moment_1', 'hu_moment_2',
        'hu_moment_3', 'hu_moment_4', 'hu_moment_5', 'hu_moment_6',
        'bbox_width', 'bbox_height', 'bbox_aspect_ratio'
    ]

    results = []
    total = 0
    errors = 0

    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            total += 1

            # Get image path based on channel
            if channel == 'CO1':
                image_path = row.get('CO1', '')
            else:  # CO5
                image_path = row.get('CO5', '')

            if not image_path or not Path(image_path).exists():
                print(f"  [{i+1}] SKIP {row.get('plate', 'N/A')}{row.get('well', 'N/A')} - {channel} file not found")
                errors += 1
                result_row = {
                    'plate': row.get('plate', ''),
                    'well': row.get('well', ''),
                    'Kinks #': row.get('Kinks #', ''),
                }
                for fname in feature_names:
                    result_row[fname] = np.nan
                results.append(result_row)
                continue

            # Extract features
            features = extract_region_properties(image_path, channel=channel)

            # Add metadata
            result_row = {
                'plate': row.get('plate', ''),
                'well': row.get('well', ''),
                'Kinks #': row.get('Kinks #', ''),
            }
            result_row.update(features)
            results.append(result_row)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] Processing {row.get('plate', 'N/A')}{row.get('well', 'N/A')}")

    # Write output
    fieldnames = ['plate', 'well', 'Kinks #'] + feature_names
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Processed {total} samples ({errors} errors)")
    print(f"✓ Output: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract region properties baseline features from fish images'
    )
    parser.add_argument(
        '--metadata',
        required=True,
        help='Path to metadata CSV (e.g., Gronske_F0_Data_TRAIN.csv)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV path'
    )
    parser.add_argument(
        '--channel',
        choices=['CO1', 'CO5'],
        default='CO5',
        help='Which channel to use (default: CO5 brightfield)'
    )

    args = parser.parse_args()

    print(f"Extracting region properties (19 features) from {args.channel} channel...")
    process_dataset(args.metadata, args.output, channel=args.channel)
