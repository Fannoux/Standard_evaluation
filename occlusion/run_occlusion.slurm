#!/bin/bash
# Occlusion interpretability. Edit the CONFIG block, then: sbatch run_occlusion.slurm
# A RegionProps-only run needs no GPU or model weights (set METHODS="RegionProps", drop --gres).

#SBATCH --job-name=occlusion
#SBATCH --output=occlusion_%j.out
#SBATCH --error=occlusion_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

# ============================== CONFIG ==============================
VENV=/path/to/venv
CSV=/path/to/manifest.csv
MASK_DIR=/path/to/test_masks
MASK_EXT=.png
OUT=results_occlusion

METHODS="RegionProps CNN ShapeEmbed"
N_PER_CLASS=40
PREDICT=latent                          # 'latent' (probe-free) or 'probe'
PROBE_TRAIN="RegionProps=/path/to/rp_features_train.csv CNN=/path/to/cnn_features_train.csv ShapeEmbed=/path/to/se_features_train.csv"

# model weights
CNN_W=/path/to/cnn_model.pth
CNN_YAML=                                # blank -> run_info.yaml next to CNN_W
VAE_W=/path/to/vae_model.pt
SE_W=/path/to/shapeembed_model.pth
SE_NORM=fro

# source repos (blank -> ../../<repo>)
CNN_SRC=
VAE_SRC=/path/to/VAE
SE_SRC=
# ===================================================================

source "$VENV/bin/activate"
# module load cuda
cd "$(dirname "$0")"

EXTRA=""
[ "$PREDICT" = "probe" ] && EXTRA="--probe-train $PROBE_TRAIN"
[ -n "$CNN_YAML" ] && EXTRA="$EXTRA --cnn-yaml $CNN_YAML"
[ -n "$CNN_SRC" ]  && EXTRA="$EXTRA --cnn-src $CNN_SRC"
[ -n "$VAE_SRC" ]  && EXTRA="$EXTRA --vae-src $VAE_SRC"
[ -n "$SE_SRC" ]   && EXTRA="$EXTRA --se-src $SE_SRC"

python run_occlusion.py \
    --csv "$CSV" --mask-dir "$MASK_DIR" --mask-ext "$MASK_EXT" \
    --split test --n-per-class "$N_PER_CLASS" --seed 0 \
    --methods $METHODS \
    --cnn-weights "$CNN_W" --vae-weights "$VAE_W" --shapeembed-weights "$SE_W" \
    --se-normalise "$SE_NORM" \
    --predict "$PREDICT" $EXTRA \
    --subset-file "$OUT/occlusion_subset_ids.txt" \
    --out "$OUT"

echo "[done] results in $OUT/"
