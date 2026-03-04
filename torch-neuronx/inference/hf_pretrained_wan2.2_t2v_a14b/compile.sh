#!/bin/bash
# Wan2.2 T2V-A14B Compilation Script
#
# Compiles all components for the T2V-A14B MoE model:
# 1. Text encoder (UMT5, TP=4, world_size=8)
# 2. Transformer (high-noise expert, TP=4, CP=2)
# 3. Transformer_2 (low-noise expert, TP=4, CP=2)
# 4. VAE decoder (NoCache, bfloat16)
# 5. Post-quant conv (float32)
#
# Run this on a Trainium2 instance with 8 Neuron cores (trn2.48xlarge).
#
# Usage:
#   ./compile.sh                        # Use default directories
#   ./compile.sh /path/to/output        # Custom output directory

set -e

# Set PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH

# Copy custom autoencoder_kl_wan.py to fix 'nearest-exact' interpolation issue on Trainium2
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
if [ -f autoencoder_kl_wan.py ]; then
    echo "Copying custom autoencoder_kl_wan.py to ${DIFFUSERS_PATH}/models/autoencoders/"
    cp autoencoder_kl_wan.py "${DIFFUSERS_PATH}/models/autoencoders/"
fi

# Configuration
COMPILED_MODELS_DIR="${1:-/opt/dlami/nvme/compiled_models_t2v_a14b}"
COMPILER_WORKDIR="${2:-/opt/dlami/nvme/compiler_workdir_t2v_a14b}"
CACHE_DIR="/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"

# Video settings (480P, should match inference)
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
MAX_SEQUENCE_LENGTH=512

# Parallelism
TP_DEGREE=4
WORLD_SIZE=8  # TP=4 x CP=2

echo "=============================================="
echo "Wan2.2 T2V-A14B Compilation"
echo "=============================================="
echo "Output: ${COMPILED_MODELS_DIR}"
echo "Compiler workdir: ${COMPILER_WORKDIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Resolution: ${HEIGHT}x${WIDTH}, Frames: ${NUM_FRAMES}"
echo "TP degree: ${TP_DEGREE}, World size: ${WORLD_SIZE}"
echo "=============================================="

# Create directories
mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

# Step 1: Cache HuggingFace model (~126GB download)
echo ""
echo "[Step 1/6] Caching HuggingFace model..."
python neuron_wan2_2_t2v_a14b/cache_hf_model.py --cache_dir "${CACHE_DIR}"

# Step 2: Compile Text Encoder (TP=4, world_size=8)
echo ""
echo "[Step 2/6] Compiling Text Encoder (TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_text_encoder.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}"

# Step 3: Compile Transformer (high-noise expert, TP=4, CP=2)
echo ""
echo "[Step 3/6] Compiling Transformer (high-noise expert, TP=${TP_DEGREE}, CP=2)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_transformer.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/transformer" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}" \
    --transformer_subfolder transformer \
    --output_dir "${COMPILED_MODELS_DIR}/transformer"

# Step 4: Compile Transformer_2 (low-noise expert, same architecture, different weights)
echo ""
echo "[Step 4/6] Compiling Transformer_2 (low-noise expert, TP=${TP_DEGREE}, CP=2)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_transformer.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/transformer_2" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}" \
    --transformer_subfolder transformer_2 \
    --output_dir "${COMPILED_MODELS_DIR}/transformer_2"

# Step 5: Compile VAE Decoder (NoCache, bfloat16)
echo ""
echo "[Step 5/6] Compiling VAE Decoder (NoCache, bfloat16, world_size=${WORLD_SIZE})..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_decoder_nocache.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/decoder" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --decoder_frames 2 \
    --tp_degree ${WORLD_SIZE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}"

# Step 6: Compile post_quant_conv (float32)
echo ""
echo "[Step 6/6] Compiling post_quant_conv (float32)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_decoder_nocache.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/pqc" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --tp_degree ${WORLD_SIZE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}" \
    --compile_post_quant_conv

echo ""
echo "=============================================="
echo "Compilation Complete!"
echo "=============================================="
echo "Models saved to: ${COMPILED_MODELS_DIR}"
echo ""
echo "Components compiled:"
echo "  - text_encoder/    (UMT5, TP=4)"
echo "  - transformer/     (high-noise expert, TP=4, CP=2)"
echo "  - transformer_2/   (low-noise expert, TP=4, CP=2)"
echo "  - decoder_nocache/ (VAE decoder, bfloat16)"
echo "  - post_quant_conv/ (float32)"
echo ""
echo "To run inference:"
echo "  NEURON_RT_NUM_CORES=8 python run_wan2.2_t2v_a14b.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo "=============================================="
