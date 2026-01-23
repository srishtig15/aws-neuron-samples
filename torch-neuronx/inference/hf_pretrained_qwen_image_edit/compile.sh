#!/bin/bash

# Compile Qwen-Image-Edit-2509 for Neuron (trn2)
# ALL components must be compiled to run on Trainium2
#
# Default settings:
#   - Image size: 512x512
#   - max_sequence_length: 512
#   - tp_degree: 8 (for transformer and language model)
#   - patch_multiplier: 2 (for image editing mode)

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH
COMPILED_MODELS_DIR="compiled_models"
COMPILER_WORKDIR="compiler_workdir"

# Parse arguments
HEIGHT=${1:-512}
WIDTH=${2:-512}
IMAGE_SIZE=${3:-224}  # Vision encoder image size (must be divisible by 14 and result in even grid)
TP_DEGREE=${4:-8}
MAX_SEQ_LEN=${5:-512}
PATCH_MULTIPLIER=${6:-2}  # 2 for image editing, 1 for generation

echo "============================================"
echo "Qwen-Image-Edit-2509 Compilation for Neuron"
echo "============================================"
echo "Image Height: ${HEIGHT}"
echo "Image Width: ${WIDTH}"
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
echo "Using modified VAE with 'nearest' interpolation (Neuron doesn't support 'nearest-exact')"
python neuron_qwen_image_edit/compile_vae.py \
    --height ${HEIGHT} \
    --width ${WIDTH} \
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

# Step 4: Text Encoder (Vision Encoder + Language Model)
echo "[Step 4/4] Compiling Text Encoder..."
echo "Note: Text encoder (Qwen2.5-VL) has two components:"
echo "  - Vision Encoder (32 blocks) - single device"
echo "  - Language Model (28 layers, TP=${TP_DEGREE})"
echo "  Using --use_subprocess to avoid XLA initialization conflicts"
python neuron_qwen_image_edit/compile_text_encoder.py \
    --mode separate \
    --use_subprocess \
    --image_size ${IMAGE_SIZE} \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --tp_degree ${TP_DEGREE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
echo "Text Encoder compiled!"
echo ""

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/"
echo "  - vae_decoder/"
echo "  - transformer/"
echo "  - vision_encoder/"
echo "  - language_model/"
echo ""
echo "To run inference on Trainium2:"
echo "  python run_qwen_image_edit.py \\"
echo "      --images input.jpg \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --compiled_height ${HEIGHT} \\"
echo "      --compiled_width ${WIDTH} \\"
echo "      --patch_multiplier ${PATCH_MULTIPLIER} \\"
echo "      --max_sequence_length ${MAX_SEQ_LEN}"
echo ""

# # batch_size=1 (无 CFG，默认)                                                                                                                                                                                           
# python run_qwen_image_edit.py --images image1.png --prompt "把女生变成男生" --transformer_batch_size 1                                                                                                                  
                                                                                                                                                                                                                        
# # batch_size=2 (有 CFG，需要用 batch_size=2 编译 transformer)                                                                                                                                                           
# python run_qwen_image_edit.py --images image1.png --prompt "把女生变成男生" --transformer_batch_size 2 --guidance_scale 7.5

# # batch_size=1 (无 CFG，默认) 多图输入示例
# python run_qwen_image_edit.py --images image1.png image2.png --prompt "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。" --transformer_batch_size 1                                                                                                                  
