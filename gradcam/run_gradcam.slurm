#!/bin/bash
# Grad-CAM cross-architecture visualization. Edit the CONFIG block, then: sbatch run_gradcam.slurm

#SBATCH --job-name=gradcam
#SBATCH --output=gradcam_%j.out
#SBATCH --error=gradcam_%j.err
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
OUT=results_gradcam

METHODS="CNN VAE ShapeEmbed"
N_VIS=3

# reuse the occlusion subset so both analyses use the same fish
SUBSET=../occlusion/results_occlusion/occlusion_subset_ids.txt

# model weights
CNN_W=/path/to/cnn_model.pth
CNN_YAML=/path/to/cnn_run_info.yaml     # blank -> run_info.yaml next to CNN_W
VAE_W=/path/to/vae_model.pt
SE_W=/path/to/shapeembed_model.pth
SE_NORM=fro

# source repos (blank -> ../../<repo>)
CNN_SRC=/path/to/scripts_cnn
VAE_SRC=/path/to/VAE
SE_SRC=/path/to/ShapeEmbedLite

# severity head fit from the train split; ShapeEmbed also needs the train masks
TRAIN_SPLIT=training
TRAIN_N_PER_CLASS=80
TRAIN_MASK_DIR=/path/to/train_masks
PROBE_TRAIN=""                          # fallback CSV, only if train re-extraction is too small
# ===================================================================

source "$VENV/bin/activate"
# module load cuda
cd "$(dirname "$0")"

EXTRA=""
[ -n "$CNN_YAML" ]       && EXTRA="$EXTRA --cnn-yaml $CNN_YAML"
[ -n "$CNN_SRC" ]        && EXTRA="$EXTRA --cnn-src $CNN_SRC"
[ -n "$VAE_SRC" ]        && EXTRA="$EXTRA --vae-src $VAE_SRC"
[ -n "$SE_SRC" ]         && EXTRA="$EXTRA --se-src $SE_SRC"
[ -n "$TRAIN_MASK_DIR" ] && EXTRA="$EXTRA --train-mask-dir $TRAIN_MASK_DIR"
[ -n "$PROBE_TRAIN" ]    && EXTRA="$EXTRA --probe-train $PROBE_TRAIN"

python run_gradcam.py \
    --csv "$CSV" --mask-dir "$MASK_DIR" --mask-ext "$MASK_EXT" \
    --split test --n-vis "$N_VIS" \
    --subset-file "$SUBSET" \
    --methods $METHODS \
    --cnn-weights "$CNN_W" --vae-weights "$VAE_W" --shapeembed-weights "$SE_W" \
    --se-normalise "$SE_NORM" \
    --train-split "$TRAIN_SPLIT" --train-n-per-class "$TRAIN_N_PER_CLASS" \
    $EXTRA \
    --out "$OUT"

echo "[done] results in $OUT/"
