# RegionProps baseline

`baseline_regionprops_masks.py` extracts the 19 scikit-image shape descriptors (the ShapeEmbed
comparison baseline) directly from binary segmentation masks. For each mask it keeps the largest
connected component (the larva) and computes: area, convex area, perimeter, major/minor axis
lengths, extent, eccentricity, solidity, feret diameter, the 7 Hu moments, and bounding-box
width/height/aspect (19 features total).

Usage:

    python baseline_regionprops_masks.py --masks <folder of mask PNGs> --output regionprops.csv

The output CSV (one row per mask, keyed by filename `stem`) plugs into the shared evaluation via
`features_to_csv.py --in-csv <that csv> --id-col stem --label-col <label> ...`.
