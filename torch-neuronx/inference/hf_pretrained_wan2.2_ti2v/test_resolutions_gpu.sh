#!/bin/bash
# Wan2.2 TI2V GPU Benchmark - Multi-Resolution
#
# Mirrors test_resolutions.sh (Trainium2) for H100 vs Trainium2 comparison.
# All parameters (prompt, negative_prompt, seed, guidance_scale, steps, etc.) are
# aligned with the Trainium2 version.
#
# Usage:
#   ./test_resolutions_gpu.sh                    # Run all configs
#   ./test_resolutions_gpu.sh --skip-warmup      # Skip warmup (faster, less accurate timing)

set -e

RESULTS_FILE="test_results_gpu.txt"
SKIP_WARMUP=false

if [[ "$1" == "--skip-warmup" ]]; then
    SKIP_WARMUP=true
fi

# Parameters (aligned with Trainium2 test_resolutions.sh / run_wan2.2_ti2v.py)
PROMPT="A cat walks on the grass, realistic"
NEGATIVE_PROMPT="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
NUM_STEPS=50
GUIDANCE_SCALE=5.0
MAX_SEQ_LEN=512
SEED=42

# Define test configurations: "HEIGHT WIDTH NUM_FRAMES FPS"
# Same as Trainium2 test_resolutions.sh
CONFIGS=(
    "384 512 81 16"    # 512x384 16fps 5s (81 frames)
    "384 512 121 24"   # 512x384 24fps 5s (121 frames)
    "480 640 81 16"    # 640x480 16fps 5s
    "480 640 121 24"   # 640x480 24fps 5s
    "704 1280 81 16"   # 1280x704 16fps 5s (720P)
    "704 1280 121 24"  # 1280x704 24fps 5s (720P)
)

echo "=============================================="
echo "Wan2.2 TI2V GPU Benchmark"
echo "=============================================="
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "Configs: ${#CONFIGS[@]}"
echo "Steps: ${NUM_STEPS}, Guidance: ${GUIDANCE_SCALE}"
echo "Results: ${RESULTS_FILE}"
echo "=============================================="

# Write results header
cat > "${RESULTS_FILE}" << EOF
Wan2.2 TI2V GPU Benchmark Results
GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')
Date: $(date)
Steps: ${NUM_STEPS}, Guidance: ${GUIDANCE_SCALE}, Seed: ${SEED}
==========================================
EOF
printf "%-12s %-8s %-8s %-12s %-12s %-10s\n" \
    "Resolution" "FPS" "Frames" "Warmup(s)" "Inference(s)" "Status" >> "${RESULTS_FILE}"
echo "----------------------------------------------------------" >> "${RESULTS_FILE}"

# Create the GPU inference Python script inline
GPU_SCRIPT=$(mktemp /tmp/wan22_gpu_bench_XXXX.py)
cat > "${GPU_SCRIPT}" << 'PYEOF'
"""Wan2.2 TI2V GPU single-run inference for benchmarking."""
import argparse
import random
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Load model (once, kept in GPU memory)
    if not hasattr(main, '_pipe'):
        print("Loading model...")
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=HUGGINGFACE_CACHE_DIR,
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=HUGGINGFACE_CACHE_DIR,
        )
        pipe = pipe.to("cuda")
        main._pipe = pipe
        print("Model loaded.")
    else:
        pipe = main._pipe

    common_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
    )

    # Warmup
    if args.do_warmup:
        print(f"\nWarmup: {args.width}x{args.height}, {args.num_frames}f, {args.num_inference_steps} steps...")
        warmup_gen = torch.Generator(device="cuda").manual_seed(args.seed + 1000)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = pipe(**common_kwargs, generator=warmup_gen).frames[0]
        torch.cuda.synchronize()
        warmup_time = time.time() - t0
        print(f"Warmup time: {warmup_time:.2f}s")
    else:
        warmup_time = 0.0

    # Main inference
    print(f"\nInference: {args.width}x{args.height}, {args.num_frames}f, {args.num_inference_steps} steps...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    t0 = time.time()
    output = pipe(**common_kwargs, generator=generator).frames[0]
    torch.cuda.synchronize()
    inference_time = time.time() - t0

    per_step = inference_time / args.num_inference_steps
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Per step: {per_step:.2f}s")
    print(f"Output frames: {len(output)}")

    # Save video
    export_to_video(output, args.output, fps=args.fps)
    print(f"Video saved: {args.output}")

    # Print machine-readable result line
    print(f"RESULT: warmup={warmup_time:.2f} inference={inference_time:.2f} status=ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output", type=str, default="output_gpu.mp4")
    parser.add_argument("--do_warmup", action="store_true")
    args = parser.parse_args()
    main(args)
PYEOF

echo "GPU inference script: ${GPU_SCRIPT}"

for config in "${CONFIGS[@]}"; do
    read -r HEIGHT WIDTH NUM_FRAMES FPS <<< "$config"
    TAG="${WIDTH}x${HEIGHT}_${FPS}fps_${NUM_FRAMES}f"

    echo ""
    echo "=============================================="
    echo "Testing: ${WIDTH}x${HEIGHT}, ${FPS}fps, ${NUM_FRAMES} frames"
    echo "=============================================="

    WARMUP_FLAG=""
    if [[ "$SKIP_WARMUP" == false ]]; then
        WARMUP_FLAG="--do_warmup"
    fi

    OUTPUT_FILE="output_gpu_${TAG}.mp4"
    LOG_FILE="log_gpu_${TAG}.txt"

    if python "${GPU_SCRIPT}" \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --num_frames ${NUM_FRAMES} \
        --prompt "${PROMPT}" \
        --negative_prompt "${NEGATIVE_PROMPT}" \
        --num_inference_steps ${NUM_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --seed ${SEED} \
        --fps ${FPS} \
        --output "${OUTPUT_FILE}" \
        ${WARMUP_FLAG} 2>&1 | tee "${LOG_FILE}"; then

        # Parse result from log
        RESULT_LINE=$(grep "^RESULT:" "${LOG_FILE}" | tail -1)
        WARMUP_TIME=$(echo "$RESULT_LINE" | grep -oP 'warmup=\K[0-9.]+')
        INFER_TIME=$(echo "$RESULT_LINE" | grep -oP 'inference=\K[0-9.]+')
        STATUS="ok"
    else
        if grep -q "out of memory\|OutOfMemoryError\|CUDA OOM" "${LOG_FILE}" 2>/dev/null; then
            STATUS="OOM"
        else
            STATUS="FAIL"
        fi
        WARMUP_TIME="-"
        INFER_TIME="-"
    fi

    printf "%-12s %-8s %-8s %-12s %-12s %-10s\n" \
        "${WIDTH}x${HEIGHT}" "${FPS}" "${NUM_FRAMES}" \
        "${WARMUP_TIME}" "${INFER_TIME}" "${STATUS}" >> "${RESULTS_FILE}"

    echo "[${TAG}] Inference: ${STATUS} ${INFER_TIME}s"
done

# Cleanup
rm -f "${GPU_SCRIPT}"

echo ""
echo "=============================================="
echo "All tests complete. Results:"
echo "=============================================="
cat "${RESULTS_FILE}"
