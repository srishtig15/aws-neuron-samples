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
    "704 1280 81 16"   # 1280x704 16fps 5s (720P, ~16.7M instructions)
    "704 1280 121 24"  # 1280x704 24fps 5s (720P)
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
TEXT_ENCODER_DIR="${COMPILED_MODELS_BASE}/text_encoder"
if [[ "$SKIP_COMPILE" == false ]] && [[ ! -d "${TEXT_ENCODER_DIR}" ]]; then
    echo ""
    echo "[Shared] Compiling Text Encoder (TP=${TP_DEGREE}, world_size=${WORLD_SIZE})..."
    mkdir -p "${COMPILED_MODELS_BASE}"
    python neuron_wan2_2_ti2v/compile_text_encoder.py \
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
    if [[ ! -e "${COMPILED_DIR}/text_encoder" ]]; then
        ln -sf "${TEXT_ENCODER_DIR}" "${COMPILED_DIR}/text_encoder"
    fi

    COMPILE_START=$(date +%s)

    # Compile transformer (skip if already exists)
    if [[ -d "${COMPILED_DIR}/transformer" ]] && [[ "$SKIP_COMPILE" == false ]]; then
        echo "[${TAG}] Transformer already compiled, skipping."
    elif [[ "$SKIP_COMPILE" == false ]]; then
        echo "[${TAG}] Compiling Transformer..."
        python neuron_wan2_2_ti2v/compile_transformer.py \
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
    # For resolutions exceeding per-operator instruction limit (720P+), use tiled spatial decoder:
    #   compile a small tile-size decoder, then tile at inference time with overlap blending.
    LATENT_H=$(( HEIGHT / 16 ))
    LATENT_W=$(( WIDTH / 16 ))
    LATENT_PIXELS=$(( LATENT_H * LATENT_W ))

    # Tiled decode threshold: 720P (44x80=3520) hits per-operator limit (NCC_EXTP003)
    USE_TILED=false
    TILE_H=384    # tile resolution in pixels
    TILE_W=512
    OVERLAP=4     # overlap in latent pixels
    if [[ ${LATENT_PIXELS} -gt 2000 ]]; then
        USE_TILED=true
    fi

    if [[ "$USE_TILED" == true ]]; then
        # Tiled decoder: compile for small tile resolution, save as decoder_tiled
        if [[ -d "${COMPILED_DIR}/decoder_tiled" ]] && [[ "$SKIP_COMPILE" == false ]]; then
            echo "[${TAG}] Tiled decoder already compiled, skipping."
        elif [[ "$SKIP_COMPILE" == false ]]; then
            echo "[${TAG}] Compiling Tiled Decoder (tile=${TILE_W}x${TILE_H}, overlap=${OVERLAP})..."
            TILE_LATENT_PIXELS=$(( (TILE_H / 16) * (TILE_W / 16) ))
            INST_LIMIT_ARG=""
            if [[ ${TILE_LATENT_PIXELS} -gt 800 ]]; then
                EST_INSTRUCTIONS=$(( TILE_LATENT_PIXELS * 6500 * 130 / 100 ))
                INST_LIMIT_ARG="--max_instruction_limit ${EST_INSTRUCTIONS}"
            fi
            # Compile decoder at tile resolution (non-stateful: tiled decode manages per-tile caches)
            python neuron_wan2_2_ti2v/compile_decoder_rolling.py \
                --compiled_models_dir "${COMPILED_DIR}" \
                --compiler_workdir "${COMPILER_WD}" \
                --height ${TILE_H} \
                --width ${TILE_W} \
                --num_frames ${NUM_FRAMES} \
                --decoder_frames 2 \
                --tp_degree ${WORLD_SIZE} \
                --world_size ${WORLD_SIZE} \
                --skip_pqc \
                --no_stateful \
                ${INST_LIMIT_ARG} 2>&1 | tee "log_compile_decoder_${TAG}.txt"
            # Rename to decoder_tiled and add tiling config
            if [[ -d "${COMPILED_DIR}/decoder_rolling" ]]; then
                mv "${COMPILED_DIR}/decoder_rolling" "${COMPILED_DIR}/decoder_tiled"
                # Update config with tiling parameters
                python3 -c "
import json
config_path = '${COMPILED_DIR}/decoder_tiled/config.json'
with open(config_path) as f:
    config = json.load(f)
config['overlap_latent'] = ${OVERLAP}
config['tiled'] = True
config['target_height'] = ${HEIGHT}
config['target_width'] = ${WIDTH}
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f'Tiled config saved: tile={config[\"height\"]}x{config[\"width\"]}, overlap=${OVERLAP}')
"
            fi
        fi
        # Compile post_quant_conv at full target resolution (separate from tiled decoder)
        if [[ ! -d "${COMPILED_DIR}/post_quant_conv" ]] && [[ "$SKIP_COMPILE" == false ]]; then
            echo "[${TAG}] Compiling post_quant_conv at full resolution (${HEIGHT}x${WIDTH})..."
            python neuron_wan2_2_ti2v/compile_decoder_rolling.py \
                --compiled_models_dir "${COMPILED_DIR}" \
                --compiler_workdir "${COMPILER_WD}" \
                --height ${HEIGHT} \
                --width ${WIDTH} \
                --num_frames ${NUM_FRAMES} \
                --tp_degree ${WORLD_SIZE} \
                --world_size ${WORLD_SIZE} \
                --skip_decoder 2>&1 | tee -a "log_compile_decoder_${TAG}.txt"
        fi
    else
        # Direct decoder: compile at full resolution
        if [[ -d "${COMPILED_DIR}/decoder_rolling" ]] && [[ "$SKIP_COMPILE" == false ]]; then
            echo "[${TAG}] Decoder already compiled, skipping."
        elif [[ "$SKIP_COMPILE" == false ]]; then
            INST_LIMIT_ARG=""
            if [[ ${LATENT_PIXELS} -gt 800 ]]; then
                EST_INSTRUCTIONS=$(( LATENT_PIXELS * 6500 * 130 / 100 ))
                INST_LIMIT_ARG="--max_instruction_limit ${EST_INSTRUCTIONS}"
                echo "[${TAG}] Compiling Decoder (non-TP rolling, latent=${LATENT_PIXELS}px, max_inst=${EST_INSTRUCTIONS})..."
            else
                echo "[${TAG}] Compiling Decoder (non-TP rolling, latent=${LATENT_PIXELS}px)..."
            fi
            python neuron_wan2_2_ti2v/compile_decoder_rolling.py \
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
    fi

    COMPILE_END=$(date +%s)
    COMPILE_TIME="$((COMPILE_END - COMPILE_START))s"

    # Run inference
    echo "[${TAG}] Running inference..."
    python run_wan2.2_ti2v.py \
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
