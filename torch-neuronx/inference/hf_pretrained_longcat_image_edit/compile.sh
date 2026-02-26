#!/bin/bash

# Compile LongCat-Image-Edit for Neuron (trn2.48xlarge)
#
# Components:
#   1. VAE: 2D AutoencoderKL (standard FLUX VAE)
#   2. Transformer: FLUX-style with TP=4, CP=2 (10 dual + 20 single stream blocks)
#   3. Vision Encoder: Qwen2.5-VL ViT with TP=4 (same as Qwen reference)
#   4. Language Model: Qwen2.5-VL LM with TP=4 (same as Qwen reference)
#
# Usage:
#   ./compile.sh                    # Compile all with defaults
#   ./compile.sh 1024 1024 448 512  # Custom dimensions

set -e

export PYTHONPATH=`pwd`/neuron_longcat_image_edit:$PYTHONPATH
COMPILED_MODELS_DIR="/opt/dlami/nvme/compiled_models"
COMPILER_WORKDIR="/opt/dlami/nvme/compiler_workdir"

# VAE compiled for full output size (no tiling needed, avoids seam artifacts)
VAE_TILE_SIZE=1024

# Parse arguments
HEIGHT=${1:-1024}
WIDTH=${2:-1024}
IMAGE_SIZE=${3:-448}
MAX_SEQ_LEN=${4:-1024}
BATCH_SIZE=${5:-1}

echo "============================================"
echo "LongCat-Image-Edit Compilation for Neuron"
echo "============================================"
echo "Output Size: ${HEIGHT}x${WIDTH}"
echo "VAE Tile Size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}"
echo "Vision Encoder Image Size: ${IMAGE_SIZE}"
echo "Max Sequence Length: ${MAX_SEQ_LEN}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Transformer: FLUX-style, TP=4, CP=2 (world_size=8)"
echo ""

# Step 1: Download model and install dependencies
echo "[Step 1/5] Downloading model and installing dependencies..."
pip install -r requirements.txt --quiet
python neuron_longcat_image_edit/cache_hf_model.py
echo "Model downloaded successfully!"
echo ""

# Step 2: Compile VAE (single device, ~5 min)
echo "[Step 2/5] Compiling VAE (2D AutoencoderKL)..."
echo "  Tile size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}"
python neuron_longcat_image_edit/compile_vae.py \
    --height ${VAE_TILE_SIZE} \
    --width ${VAE_TILE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "VAE compiled!"
echo ""

# Step 3: Compile Transformer (TP=4, CP=2, ~30-60 min)
echo "[Step 3/5] Compiling FLUX Transformer (TP=4, CP=2)..."
neuron_parallel_compile python neuron_longcat_image_edit/compile_transformer.py \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --tp_degree 4 \
    --world_size 8 \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Transformer compiled!"
echo ""

# Step 4: Compile Vision Encoder (TP=4, ~10 min)
echo "[Step 4/5] Compiling Vision Encoder (TP=4, float32)..."
python neuron_longcat_image_edit/compile_vision_encoder.py \
    --image_size ${IMAGE_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Vision Encoder compiled!"
echo ""

# Step 5: Compile Language Model (TP=4, ~15 min)
echo "[Step 5/5] Compiling Language Model (TP=4)..."
neuron_parallel_compile python neuron_longcat_image_edit/compile_language_model.py \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Language Model compiled!"
echo ""

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - vae_decoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - transformer/ (TP=4, CP=2, output: ${HEIGHT}x${WIDTH})"
echo "  - vision_encoder/ (TP=4, float32)"
echo "  - language_model/ (TP=4)"
echo ""
echo "To run inference:"
echo "  NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \\"
echo "      --image input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --warmup"
