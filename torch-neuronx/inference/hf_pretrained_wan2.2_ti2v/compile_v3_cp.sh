#!/bin/bash
# Wan2.2 V3 CP Compilation Script
#
# This script compiles the Wan2.2 transformer with Context Parallel (TP=4, CP=2).
# Run this on a Trainium2 instance with 8 Neuron cores.
#
# Usage:
#   ./compile_v3_cp.sh                    # Use default directories
#   ./compile_v3_cp.sh /path/to/output    # Custom output directory

set -e

# Set PYTHONPATH (same as compile_latency_optimized_v2.sh)
export PYTHONPATH=`pwd`:$PYTHONPATH

# Copy custom autoencoder_kl_wan.py to fix 'nearest-exact' interpolation issue on Trainium2
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
echo "Copying custom autoencoder_kl_wan.py to ${DIFFUSERS_PATH}/models/autoencoders/"
cp autoencoder_kl_wan.py "${DIFFUSERS_PATH}/models/autoencoders/"

# Configuration
COMPILED_MODELS_DIR="${1:-/opt/dlami/nvme/compiled_models_v3_cp}"
COMPILER_WORKDIR="${2:-/opt/dlami/nvme/compiler_workdir_v3_cp}"

# Video settings (should match inference)
HEIGHT=512
WIDTH=512
NUM_FRAMES=81
MAX_SEQUENCE_LENGTH=512

# Parallelism
TP_DEGREE=4
WORLD_SIZE=8  # TP=4 x CP=2

echo "=============================================="
echo "Wan2.2 V3 CP Compilation"
echo "=============================================="
echo "Output: ${COMPILED_MODELS_DIR}"
echo "Compiler workdir: ${COMPILER_WORKDIR}"
echo "Resolution: ${HEIGHT}x${WIDTH}, Frames: ${NUM_FRAMES}"
echo "TP degree: ${TP_DEGREE}, World size: ${WORLD_SIZE}"
echo "=============================================="

# Create directories
mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

# Step 1: Cache HuggingFace model (if not already cached)
echo ""
echo "[Step 1/4] Caching HuggingFace model..."
python neuron_wan2_2_ti2v/cache_hf_model.py

# Step 2: Compile Text Encoder (V2, TP=4 to match transformer)
# Note: Must use TP=4 to share parallel state with transformer (TP=4, CP=2)
# At inference time, the 4 TP checkpoints are duplicated for 2 CP ranks → 8 total
echo ""
echo "[Step 2/4] Compiling Text Encoder (V2, TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
python neuron_wan2_2_ti2v/compile_text_encoder_v2.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE}

# Step 3: Compile Transformer (V3 CP, TP=4, CP=2)
echo ""
echo "[Step 3/4] Compiling Transformer (V3 CP, TP=${TP_DEGREE}, CP=2)..."
python neuron_wan2_2_ti2v/compile_transformer_v3_cp.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE}

# Step 4: Compile Decoder and post_quant_conv (V2)
# Decoder doesn't use actual TP sharding (weights duplicated), but must match world_size
echo ""
echo "[Step 4/4] Compiling Decoder and post_quant_conv (V2, TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
# python neuron_wan2_2_ti2v/compile_decoder_v2_optimized.py \
#     --compiled_models_dir "${COMPILED_MODELS_DIR}" \
#     --height ${HEIGHT} \
#     --width ${WIDTH} \
#     --num_frames ${NUM_FRAMES} \
#     --tp_degree ${TP_DEGREE} \
#     --world_size ${WORLD_SIZE}

python neuron_wan2_2_ti2v/compile_decoder.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES}

echo ""
echo "=============================================="
echo "Compilation Complete!"
echo "=============================================="
echo "Models saved to: ${COMPILED_MODELS_DIR}"
echo ""
echo "To run inference:"
echo "  NEURON_RT_NUM_CORES=8 python run_wan2.2_ti2v_v3_cp.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --prompt 'A cat walks on the grass, realistic' \\"
echo "    --force_v1_decoder"
echo "=============================================="