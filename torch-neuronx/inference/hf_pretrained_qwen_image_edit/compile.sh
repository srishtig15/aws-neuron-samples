#!/bin/bash

# Compile Qwen-Image-Edit-2509 for Neuron (trn2)
# ALL components must be compiled to run on Trainium2
#
# Default settings:
#   - Output size: 1024x1024
#   - VAE tile size: 512x512 (fixed, uses tiled processing for larger images)
#   - max_sequence_length: 512
#   - tp_degree: 8 (for transformer)
#   - patch_multiplier: 2 (for image editing mode)

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH
COMPILED_MODELS_DIR="/opt/dlami/nvme/compiled_models"
COMPILER_WORKDIR="/opt/dlami/nvme/compiler_workdir"

# Fixed VAE tile size (VAE uses tiled processing for larger images)
VAE_TILE_SIZE=512

# Parse arguments
HEIGHT=${1:-1024}
WIDTH=${2:-1024}
IMAGE_SIZE=${3:-224}  # Vision encoder image size (must be divisible by 14 and result in even grid)
TP_DEGREE=${4:-8}
MAX_SEQ_LEN=${5:-512}
PATCH_MULTIPLIER=${6:-3}  # 2 for single image editing, 3 for 2 images merging, 1 for generation

echo "============================================"
echo "Qwen-Image-Edit-2509 Compilation for Neuron"
echo "============================================"
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
python neuron_qwen_image_edit/compile_transformer.py \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --tp_degree ${TP_DEGREE} \
    --patch_multiplier ${PATCH_MULTIPLIER} \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Transformer compiled successfully!"
echo ""

# # Step 4: Vision Encoder (Language Model runs on CPU)
# echo "[Step 4/4] Compiling Vision Encoder..."
# echo "Note: Text encoder (Qwen2.5-VL) has two components:"
# echo "  - Vision Encoder: compiled on single device (dims not divisible by TP=8)"
# echo "  - Language Model: runs on CPU (28Q/4KV heads incompatible with TP=8)"
# python neuron_qwen_image_edit/compile_text_encoder.py \
#     --vision_only \
#     --image_size ${IMAGE_SIZE} \
#     --compiled_models_dir ${COMPILED_MODELS_DIR} \
#     --compiler_workdir ${COMPILER_WORKDIR}
# echo "Vision Encoder compiled!"
# echo ""

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - vae_decoder/ (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE})"
echo "  - transformer/ (TP=${TP_DEGREE}, output: ${HEIGHT}x${WIDTH})"
echo ""
echo "Note: Language model runs on CPU (GQA 28Q/4KV incompatible with TP=8)"
echo ""
echo "To run inference on Trainium2:"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --height ${HEIGHT} \\"
echo "      --width ${WIDTH} \\"
echo "      --patch_multiplier ${PATCH_MULTIPLIER} \\"
echo "      --max_sequence_length ${MAX_SEQ_LEN}"
echo ""

# 单图编辑示例 (CFG默认开启，true_cfg_scale=4.0)
# NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png --prompt "把女生变成男生" --warmup

# 多图合成示例 (需要 patch_multiplier=3)
# NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "..." --patch_multiplier 3 --warmup

# 完整运行示例
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --patch_multiplier 3 --warmup --height 1024 --width 1024 --max_sequence_length 512
