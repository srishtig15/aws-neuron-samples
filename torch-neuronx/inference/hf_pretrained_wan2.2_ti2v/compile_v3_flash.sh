#!/bin/bash
# Wan2.2 V3 Flash Compilation Script
#
# This script compiles Wan2.2 with NKI Flash Attention (TP=8, no Context Parallel).
# Simpler than v3_cp, avoids collective operation serialization issues.
#
# Components:
# - text_encoder: TP=8, world_size=8
# - transformer: TP=8, world_size=8, NKI Flash Attention
# - decoder/post_quant_conv: world_size=8, -O1 optimization
#
# Usage:
#   ./compile_v3_flash.sh                    # Use default directories
#   ./compile_v3_flash.sh /path/to/output    # Custom output directory

set -e

# Set PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH

# Copy custom autoencoder_kl_wan.py to fix 'nearest-exact' interpolation issue on Trainium2
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
echo "Copying custom autoencoder_kl_wan.py to ${DIFFUSERS_PATH}/models/autoencoders/"
cp autoencoder_kl_wan.py "${DIFFUSERS_PATH}/models/autoencoders/"

# Configuration
COMPILED_MODELS_DIR="${1:-/opt/dlami/nvme/compiled_models_v3_flash}"
COMPILER_WORKDIR="${2:-/opt/dlami/nvme/compiler_workdir_v3_flash}"
CACHE_DIR="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"

# Video settings (should match inference)
HEIGHT=512
WIDTH=512
NUM_FRAMES=81
MAX_SEQUENCE_LENGTH=512

# Parallelism - TP=8, world_size=8 (no Context Parallel)
TP_DEGREE=8
WORLD_SIZE=8

echo "=============================================="
echo "Wan2.2 V3 Flash Compilation (NKI, TP=8)"
echo "=============================================="
echo "Output: ${COMPILED_MODELS_DIR}"
echo "Compiler workdir: ${COMPILER_WORKDIR}"
echo "Resolution: ${HEIGHT}x${WIDTH}, Frames: ${NUM_FRAMES}"
echo "Transformer: TP=${TP_DEGREE}, world_size=${WORLD_SIZE}, NKI Flash Attention"
echo "Decoder: world_size=${WORLD_SIZE}, bfloat16, --model-type=unet-inference"
echo "=============================================="

# Create directories
mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

# Step 1: Cache HuggingFace model (if not already cached)
echo ""
echo "[Step 1/4] Caching HuggingFace model..."
python neuron_wan2_2_ti2v/cache_hf_model.py

# Step 2: Compile Text Encoder (V2, TP=8, world_size=8)
echo ""
echo "[Step 2/4] Compiling Text Encoder (V2, TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
python neuron_wan2_2_ti2v/compile_text_encoder_v2.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE}

# Step 3: Compile Transformer (V3 Flash, TP=8, NKI)
echo ""
echo "[Step 3/4] Compiling Transformer (V3 Flash, TP=${TP_DEGREE})..."
python neuron_wan2_2_ti2v/compile_transformer_v3_flash.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}" \
    --cache_dir "${CACHE_DIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE}

# Step 4: Compile Decoder and post_quant_conv (V3)
# V3 uses --model-type=unet-inference (not transformer) and bfloat16 for decoder
# Decoder doesn't use actual TP sharding (weights duplicated), but must match world_size
echo ""
echo "[Step 4/4] Compiling Decoder and post_quant_conv (V3, bfloat16, world_size=${WORLD_SIZE})..."
python neuron_wan2_2_ti2v/compile_decoder_v3.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE}

# Note: VAE Encoder is NOT compiled to Neuron due to a Neuron compiler bug
# (NCC_IBIR158) in the Conv3D tensorizer at 256x256 spatial resolution.
# Wan2.2's encoder uses WanCausalConv3d (3D convolutions), unlike Qwen's Conv2D encoder.
# For I2V mode, the encoder runs on CPU (runs once per video, negligible overhead).
# The run scripts automatically fall back to CPU when encoder_v3/ is not found.

echo ""
echo "=============================================="
echo "Compilation Complete!"
echo "=============================================="
echo "Models saved to: ${COMPILED_MODELS_DIR}"
echo ""
echo "To run T2V inference:"
echo "  python run_wan2.2_ti2v_v3_flash.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo ""
echo "To run I2V inference:"
echo "  python run_wan2.2_ti2v_v3_flash.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --image input.png \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo "=============================================="
