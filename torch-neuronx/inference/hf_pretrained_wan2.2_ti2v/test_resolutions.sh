#!/bin/bash
# Test Wan2.2 TI2V across multiple resolutions and frame rates.
#
# Configurations: 512x384, 640x480, 1280x720 at 16fps and 24fps, 5s duration.
# Text encoder is compiled once and shared across all configs.
#
# Usage:
#   ./test_resolutions.sh              # Run all configs
#   ./test_resolutions.sh --skip-compile  # Skip compilation, only run inference

set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

# # Copy custom autoencoder_kl_wan.py
# DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
# cp autoencoder_kl_wan.py "${DIFFUSERS_PATH}/models/autoencoders/"

# Fix nearest-exact -> nearest for Trainium2 compatibility
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
fi

BASE_DIR="/opt/dlami/nvme"
COMPILER_WORKDIR_BASE="${BASE_DIR}/compiler_workdir_test"
COMPILED_MODELS_BASE="${BASE_DIR}/compiled_models_test"
RESULTS_FILE="test_results.txt"
SKIP_COMPILE=false

if [[ "$1" == "--skip-compile" ]]; then
    SKIP_COMPILE=true
fi

# Parallelism (same for all configs)
TP_DEGREE=4
WORLD_SIZE=8
MAX_SEQ_LEN=512

# Define test configurations: "HEIGHT WIDTH NUM_FRAMES FPS"
# Wan2.2 frame formula: frames = fps * duration + 1, must satisfy (frames-1) % 4 == 0
CONFIGS=(
    "384 512 81 16"    # 512x384 16fps 5s (81 frames)
    "384 512 121 24"   # 512x384 24fps 5s (121 frames)
    "480 640 81 16"    # 640x480 16fps 5s
    "480 640 121 24"   # 640x480 24fps 5s
    # "704 1280 81 16"   # 1280x704 16fps 5s (720P, ~16.7M instructions)
    # "704 1280 121 24"  # 1280x704 24fps 5s (720P)
)

echo "=============================================="
echo "Wan2.2 TI2V Multi-Resolution Test"
echo "=============================================="
echo "Configs: ${#CONFIGS[@]}"
echo "Results: ${RESULTS_FILE}"
echo "=============================================="

# Clear results file
echo "Wan2.2 TI2V Multi-Resolution Test Results" > "${RESULTS_FILE}"
echo "Date: $(date)" >> "${RESULTS_FILE}"
echo "==========================================" >> "${RESULTS_FILE}"
printf "%-12s %-8s %-8s %-10s %-14s %-10s\n" \
    "Resolution" "FPS" "Frames" "Compile" "Inference(s)" "Per-step(s)" >> "${RESULTS_FILE}"
echo "--------------------------------------------------------------" >> "${RESULTS_FILE}"

# Step 1: Compile text encoder once (shared across all configs)
TEXT_ENCODER_DIR="${COMPILED_MODELS_BASE}/text_encoder_v2"
if [[ "$SKIP_COMPILE" == false ]] && [[ ! -d "${TEXT_ENCODER_DIR}" ]]; then
    echo ""
    echo "[Shared] Compiling Text Encoder (TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
    mkdir -p "${COMPILED_MODELS_BASE}"
    python neuron_wan2_2_ti2v/compile_text_encoder_v2.py \
        --compiled_models_dir "${COMPILED_MODELS_BASE}" \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --tp_degree ${TP_DEGREE} \
        --world_size ${WORLD_SIZE}
    echo "Text encoder compiled."
else
    echo "[Shared] Text encoder already compiled or skipped."
fi

# Step 2: Iterate through configs
for config in "${CONFIGS[@]}"; do
    read -r HEIGHT WIDTH NUM_FRAMES FPS <<< "$config"
    TAG="${WIDTH}x${HEIGHT}_${FPS}fps_${NUM_FRAMES}f"
    COMPILED_DIR="${COMPILED_MODELS_BASE}_${TAG}"
    COMPILER_WD="${COMPILER_WORKDIR_BASE}_${TAG}"

    echo ""
    echo "=============================================="
    echo "Testing: ${WIDTH}x${HEIGHT}, ${FPS}fps, ${NUM_FRAMES} frames"
    echo "=============================================="

    # Create output dir and symlink shared text encoder
    mkdir -p "${COMPILED_DIR}"
    if [[ ! -e "${COMPILED_DIR}/text_encoder_v2" ]]; then
        ln -sf "${TEXT_ENCODER_DIR}" "${COMPILED_DIR}/text_encoder_v2"
    fi

    COMPILE_START=$(date +%s)

    # Compile transformer (skip if already exists)
    if [[ -d "${COMPILED_DIR}/transformer_v3_cp" ]] && [[ "$SKIP_COMPILE" == false ]]; then
        echo "[${TAG}] Transformer already compiled, skipping."
    elif [[ "$SKIP_COMPILE" == false ]]; then
        echo "[${TAG}] Compiling Transformer..."
        python neuron_wan2_2_ti2v/compile_transformer_v3_cp.py \
            --compiled_models_dir "${COMPILED_DIR}" \
            --compiler_workdir "${COMPILER_WD}" \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --num_frames ${NUM_FRAMES} \
            --max_sequence_length ${MAX_SEQ_LEN} \
            --tp_degree ${TP_DEGREE} \
            --world_size ${WORLD_SIZE} 2>&1 | tee "log_compile_transformer_${TAG}.txt"
    fi

    # Compile decoder (skip if already exists)
    # Always use non-TP rolling decoder for exact quality (TP sharding causes blurry output).
    # For resolutions exceeding default 5M instruction limit, raise it via --max-instruction-limit.
    if [[ -d "${COMPILED_DIR}/decoder_v3_rolling" ]] && [[ "$SKIP_COMPILE" == false ]]; then
        echo "[${TAG}] Decoder already compiled, skipping."
    elif [[ "$SKIP_COMPILE" == false ]]; then
        LATENT_PIXELS=$(( (HEIGHT / 16) * (WIDTH / 16) ))
        # Estimate instruction count: ~6.5 instructions per latent pixel (empirical)
        # 512x384 (768 px) -> ~5M, 640x480 (1200 px) -> ~5.7M, 1280x704 (3520 px) -> ~16.7M
        INST_LIMIT_ARG=""
        if [[ ${LATENT_PIXELS} -gt 800 ]]; then
            # Add 30% margin over estimated instruction count
            EST_INSTRUCTIONS=$(( LATENT_PIXELS * 6500 * 130 / 100 ))
            INST_LIMIT_ARG="--max_instruction_limit ${EST_INSTRUCTIONS}"
            echo "[${TAG}] Compiling Decoder (non-TP rolling, latent=${LATENT_PIXELS}px, max_inst=${EST_INSTRUCTIONS})..."
        else
            echo "[${TAG}] Compiling Decoder (non-TP rolling, latent=${LATENT_PIXELS}px)..."
        fi
        python neuron_wan2_2_ti2v/compile_decoder_v3_rolling.py \
            --compiled_models_dir "${COMPILED_DIR}" \
            --compiler_workdir "${COMPILER_WD}" \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --num_frames ${NUM_FRAMES} \
            --decoder_frames 2 \
            --tp_degree ${WORLD_SIZE} \
            --world_size ${WORLD_SIZE} \
            ${INST_LIMIT_ARG} 2>&1 | tee "log_compile_decoder_${TAG}.txt"
    fi

    COMPILE_END=$(date +%s)
    COMPILE_TIME="$((COMPILE_END - COMPILE_START))s"

    # Run inference
    echo "[${TAG}] Running inference..."
    python run_wan2.2_ti2v_v3_cp.py \
        --compiled_models_dir "${COMPILED_DIR}" \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --num_frames ${NUM_FRAMES} \
        --num_inference_steps 50 \
        --fps ${FPS} \
        --prompt "A cat walks on the grass, realistic" \
        --output "output_${TAG}.mp4" 2>&1 | tee "log_infer_${TAG}.txt"

    # Parse timing from Python script output (excludes model loading and warmup)
    INFER_TIME=$(grep -oP 'inference time: \K[0-9.]+' "log_infer_${TAG}.txt" | tail -1)
    PER_STEP=$(grep -oP 'Per step \(denoise only\): \K[0-9.]+' "log_infer_${TAG}.txt" | tail -1)
    INFER_TIME="${INFER_TIME:-N/A}"
    PER_STEP="${PER_STEP:-N/A}"

    # Log result
    printf "%-12s %-8s %-8s %-10s %-14s %-10s\n" \
        "${WIDTH}x${HEIGHT}" "${FPS}" "${NUM_FRAMES}" \
        "${COMPILE_TIME}" "${INFER_TIME}" "${PER_STEP}" >> "${RESULTS_FILE}"

    echo "[${TAG}] Compile: ${COMPILE_TIME}, Inference: ${INFER_TIME}s, Per-step: ${PER_STEP}s"
done

echo ""
echo "=============================================="
echo "All tests complete. Results:"
echo "=============================================="
cat "${RESULTS_FILE}"
