#!/bin/bash
set -e

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b
export PYTHONPATH=$(pwd):$PYTHONPATH

COMPILED_MODELS_DIR=/opt/dlami/nvme/compiled_models_t2v_a14b
COMPILER_WORKDIR=/opt/dlami/nvme/compiler_workdir_t2v_a14b
CACHE_DIR=/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
MAX_SEQUENCE_LENGTH=512
TP_DEGREE=4

mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

# Fix nearest-exact -> nearest for Trainium2 compatibility
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
fi

CP_DEGREE=2
WORLD_SIZE=$((TP_DEGREE * CP_DEGREE))

echo "=============================================="
echo "Wan2.2 T2V-A14B Full Compilation"
echo "TP=${TP_DEGREE}, CP=${CP_DEGREE}, Resolution: ${HEIGHT}x${WIDTH}x${NUM_FRAMES}"
echo "=============================================="

echo "[Step 1/6] Caching HuggingFace model..."
python neuron_wan2_2_t2v_a14b/cache_hf_model.py --cache_dir "${CACHE_DIR}"

echo "[Step 2/6] Compiling Text Encoder..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_text_encoder.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size ${WORLD_SIZE} \
    --cache_dir "${CACHE_DIR}"

echo "[Step 3/6] Compiling Transformer (high-noise expert)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_transformer.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/transformer" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --cp_degree ${CP_DEGREE} \
    --cache_dir "${CACHE_DIR}" \
    --transformer_subfolder transformer \
    --output_dir "${COMPILED_MODELS_DIR}/transformer"

echo "[Step 4/6] Compiling Transformer_2 (low-noise expert)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_transformer.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/transformer_2" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --tp_degree ${TP_DEGREE} \
    --cp_degree ${CP_DEGREE} \
    --cache_dir "${CACHE_DIR}" \
    --transformer_subfolder transformer_2 \
    --output_dir "${COMPILED_MODELS_DIR}/transformer_2"

echo "[Step 5/6] Compiling VAE Decoder (Rolling Cache)..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_decoder_rolling.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/decoder_rolling" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --decoder_frames 2 \
    --tp_degree 8 \
    --world_size 8 \
    --cache_dir "${CACHE_DIR}"

echo "[Step 6/6] Compiling post_quant_conv..."
WORLD_SIZE=${WORLD_SIZE} neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_decoder_nocache.py \
    --compiled_models_dir "${COMPILED_MODELS_DIR}" \
    --compiler_workdir "${COMPILER_WORKDIR}/pqc" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --tp_degree 8 \
    --world_size 8 \
    --cache_dir "${CACHE_DIR}" \
    --compile_post_quant_conv

echo "=============================================="
echo "Compilation Complete!"
echo "Models saved to: ${COMPILED_MODELS_DIR}"
echo "=============================================="
