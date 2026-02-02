#!/bin/bash

# Compile Qwen-Image-Edit-2509 for Neuron (trn2)
# ALL components must be compiled to run on Trainium2
#
# Default settings:
#   - Output size: 1024x1024
#   - VAE tile size: 512x512 (fixed, uses tiled processing for larger images)
#   - max_sequence_length: 1024
#   - tp_degree: 8 (for transformer)
#   - patch_multiplier: 3 (for 2-image merging)
#
# Usage:
#   ./compile.sh                    # Compile all versions
#   ./compile.sh v1                 # Compile V1 only
#   ./compile.sh v2                 # Compile V2 only
#   ./compile.sh v1_flash           # Compile V1 Flash only (recommended, NKI Flash Attention)
#   ./compile.sh v2_flash           # Compile V2 Flash only (ModelBuilder + NKI)
#   ./compile.sh v3_cp              # Compile V3 CP (Context Parallel + NKI, experimental)
#   ./compile.sh 1024 1024 448 8 1024 3  # Custom dimensions

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH
COMPILED_MODELS_DIR="/opt/dlami/nvme/compiled_models"
COMPILER_WORKDIR="/opt/dlami/nvme/compiler_workdir"

# Fixed VAE tile size (VAE uses tiled processing for larger images)
VAE_TILE_SIZE=512

# Check if first argument is version selector
VERSION_MODE="all"
if [[ "$1" == "v1" || "$1" == "v2" || "$1" == "v1_flash" || "$1" == "v2_flash" || "$1" == "v3_cp" ]]; then
    VERSION_MODE="$1"
    shift
fi

# Parse arguments
HEIGHT=${1:-1024}
WIDTH=${2:-1024}
IMAGE_SIZE=${3:-448}  # Vision encoder image size (must be divisible by 14 and result in even grid)
TP_DEGREE=${4:-8}
MAX_SEQ_LEN=${5:-1024}
PATCH_MULTIPLIER=${6:-3}  # 2 for single image editing, 3 for 2 images merging, 1 for generation

echo "============================================"
echo "Qwen-Image-Edit-2509 Compilation for Neuron"
echo "============================================"
echo "Transformer Version: ${VERSION_MODE}"
echo "Output Size: ${HEIGHT}x${WIDTH}"
echo "VAE Tile Size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE} (fixed)"
echo "Vision Encoder Image Size: ${IMAGE_SIZE}"
echo "TP Degree: ${TP_DEGREE}"
echo "Max Sequence Length: ${MAX_SEQ_LEN}"
echo "Patch Multiplier: ${PATCH_MULTIPLIER}"
echo ""

# Step 1: Download the model
echo "[Step 1/4] Downloading model..."
python neuron_qwen_image_edit/cache_hf_model.py
echo "Model downloaded successfully!"
echo ""

# Step 2: Compile VAE (encoder and decoder)
echo "[Step 2/4] Compiling VAE..."
echo "VAE tile size: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE} (tiled processing for larger images)"
echo "Using modified VAE with 'nearest' interpolation (Neuron doesn't support 'nearest-exact')"
python neuron_qwen_image_edit/compile_vae.py \
    --height ${VAE_TILE_SIZE} \
    --width ${VAE_TILE_SIZE} \
    --temporal_frames 1 \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "VAE compiled successfully!"
echo ""

# Step 3: Compile Transformer
echo "[Step 3/4] Compiling Transformer..."
echo "  TP=${TP_DEGREE}, patch_multiplier=${PATCH_MULTIPLIER} (for image editing)"

if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v1" ]]; then
    echo "  Compiling V1 (parallel_model_trace)..."
    python neuron_qwen_image_edit/compile_transformer.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree ${TP_DEGREE} \
        --patch_multiplier ${PATCH_MULTIPLIER} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "  V1 Transformer compiled successfully!"
fi

if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v2" ]]; then
    echo "  Compiling V2 (ModelBuilder)..."
    python neuron_qwen_image_edit/compile_transformer_v2.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree ${TP_DEGREE} \
        --patch_multiplier ${PATCH_MULTIPLIER} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR}
    echo "  V2 Transformer compiled successfully!"
fi

if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v1_flash" ]]; then
    echo "  Compiling V1 Flash (NKI Flash Attention, recommended)..."
    python neuron_qwen_image_edit/compile_transformer_v1_flash.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree ${TP_DEGREE} \
        --patch_multiplier ${PATCH_MULTIPLIER} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "  V1 Flash Transformer compiled successfully!"
fi

if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v2_flash" ]]; then
    echo "  Compiling V2 Flash (ModelBuilder + NKI Flash Attention)..."
    python neuron_qwen_image_edit/compile_transformer_v2_flash.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree ${TP_DEGREE} \
        --patch_multiplier ${PATCH_MULTIPLIER} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR}
    echo "  V2 Flash Transformer compiled successfully!"
fi

if [[ "$VERSION_MODE" == "v3_cp" ]]; then
    echo "  Compiling V3 CP (Context Parallel + NKI Flash Attention)..."
    echo "  Using TP=4, world_size=8 (CP=2)"
    python neuron_qwen_image_edit/compile_transformer_v3_cp.py \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        --tp_degree 4 \
        --world_size 8 \
        --patch_multiplier ${PATCH_MULTIPLIER} \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "  V3 CP Transformer compiled successfully!"

    # Also compile V3 Language Model (ModelBuilder API, TP=4, world_size=8)
    echo ""
    echo "  Compiling V3 Language Model (ModelBuilder API)..."
    echo "  Using TP=4, world_size=8 (compatible with V3 CP transformer)"
    python neuron_qwen_image_edit/compile_language_model_v3.py \
        --max_sequence_length ${MAX_SEQ_LEN} \
        --compiled_models_dir ${COMPILED_MODELS_DIR} \
        --compiler_workdir ${COMPILER_WORKDIR}
    echo "  V3 Language Model compiled successfully!"
fi
echo ""

# Step 4: Vision Encoder
echo "[Step 4/4] Compiling Vision Encoder..."
echo "Note: Text encoder (Qwen2.5-VL) has two components:"
echo "  - Vision Encoder: compiled on single device (dims not divisible by TP=8)"
if [[ "$VERSION_MODE" == "v3_cp" ]]; then
    echo "  - Language Model: compiled with V3 API (TP=4, world_size=8)"
else
    echo "  - Language Model: runs on CPU (28Q/4KV heads incompatible with TP=8)"
fi
python neuron_qwen_image_edit/compile_text_encoder.py \
    --vision_only \
    --image_size ${IMAGE_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Vision Encoder compiled!"
echo ""

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - vae_decoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v1" ]]; then
    echo "  - transformer/ (V1, TP=${TP_DEGREE}, output: ${HEIGHT}x${WIDTH})"
fi
if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v2" ]]; then
    echo "  - transformer_v2/ (V2, TP=${TP_DEGREE}, output: ${HEIGHT}x${WIDTH})"
fi
if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v1_flash" ]]; then
    echo "  - transformer_v1_flash/ (V1 Flash, TP=${TP_DEGREE}, output: ${HEIGHT}x${WIDTH}, NKI Flash Attention)"
fi
if [[ "$VERSION_MODE" == "all" || "$VERSION_MODE" == "v2_flash" ]]; then
    echo "  - transformer_v2_flash/ (V2 Flash, TP=${TP_DEGREE}, output: ${HEIGHT}x${WIDTH}, ModelBuilder + NKI)"
fi
if [[ "$VERSION_MODE" == "v3_cp" ]]; then
    echo "  - transformer_v3_cp/ (V3 CP, TP=4, CP=2, output: ${HEIGHT}x${WIDTH}, Context Parallel + NKI)"
    echo "  - language_model_v3/ (V3, TP=4, world_size=8, ModelBuilder API)"
fi
echo "  - vision_encoder/"
echo ""
if [[ "$VERSION_MODE" == "v3_cp" ]]; then
    echo "Note: Language model compiled with V3 API (TP=4, world_size=8)"
    echo "      Compatible with V3 CP transformer"
else
    echo "Note: Language model runs on CPU (GQA 28Q/4KV incompatible with TP=8)"
fi
echo ""
echo "To run inference on Trainium2:"
echo ""
if [[ "$VERSION_MODE" == "v3_cp" ]]; then
    echo "  # V3 CP (Context Parallel + NKI, fastest, with V3 Language Model on Neuron):"
    echo "  NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \\"
    echo "      --images input.jpg \\"
    echo "      --prompt \"your edit instruction\" \\"
    echo "      --use_v3_cp --use_v3_language_model"
    echo ""
fi
echo "  # V1 Flash (recommended, NKI Flash Attention):"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --use_v1_flash"
echo ""
echo "  # V2 Flash (ModelBuilder + NKI, same speed as V1 Flash):"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --use_v2_flash"
echo ""
echo "  # V2 (ModelBuilder):"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --use_v2"
echo ""
echo "  # V1:"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\""
echo ""

# 单图编辑示例 (CFG默认开启，true_cfg_scale=4.0)
# NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png --prompt "把女生变成男生" --warmup

# 多图合成示例 (需要 patch_multiplier=3)
# NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "..." --patch_multiplier 3 --warmup

# 完整运行示例
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup --use_v2
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup --use_v1_flash 
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup --use_v2_flash 
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup --use_v3_cp
