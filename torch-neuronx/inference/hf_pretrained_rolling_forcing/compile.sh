#!/bin/bash
# =============================================================================
# RollingForcing Neuron Compilation Script
# =============================================================================
# Target: trn2.48xlarge
# Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
#
# Compiles all components:
#   1. Download models (~5 min)
#   2. Text encoder (UMT5-XXL, TP=8, ~15 min)
#   3. VAE decoder (NoCache, TP=8, ~5 min)
#   4. Transformer (CausalWanModel, TP=4, ~30-60 min)
#
# Usage:
#   chmod +x compile.sh
#   ./compile.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
CACHE_DIR=/opt/dlami/nvme/rolling_forcing_hf_cache
COMPILED_DIR=compiled_models
COMPILER_DIR=compiler_workdir
CHECKPOINT_PATH="$CACHE_DIR/rolling_forcing/checkpoints/rolling_forcing_dmd.pt"

echo "============================================================"
echo "RollingForcing Neuron Compilation"
echo "============================================================"
echo "Venv: $VENV"
echo "Cache: $CACHE_DIR"
echo "Output: $COMPILED_DIR"
echo "============================================================"

# Activate venv
source $VENV/bin/activate

# Ensure RollingForcing repo is cloned
if [ ! -d "/tmp/RollingForcing" ]; then
    echo "Cloning RollingForcing repo..."
    git clone https://github.com/TencentARC/RollingForcing.git /tmp/RollingForcing
fi

# Install dependencies
pip install -q diffusers transformers omegaconf easydict einops safetensors accelerate sentencepiece protobuf imageio

# =============================================================================
# Step 1: Download models
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Downloading models..."
echo "============================================================"
python neuron_rolling_forcing/cache_hf_model.py --cache_dir $CACHE_DIR

# =============================================================================
# Step 2: Compile text encoder (UMT5-XXL, TP=8)
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Compiling text encoder (TP=8)..."
echo "============================================================"
WORLD_SIZE=8 neuron_parallel_compile python neuron_rolling_forcing/compile_text_encoder.py \
    --tp_degree 8 \
    --max_sequence_length 512 \
    --compiled_models_dir $COMPILED_DIR \
    --compiler_workdir $COMPILER_DIR \
    --cache_dir $CACHE_DIR

# =============================================================================
# Step 3: Compile VAE decoder (NoCache, TP=8)
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: Compiling VAE decoder (TP=8, NoCache)..."
echo "============================================================"
WORLD_SIZE=8 neuron_parallel_compile python neuron_rolling_forcing/compile_decoder_nocache.py \
    --tp_degree 8 \
    --world_size 8 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --compiled_models_dir $COMPILED_DIR \
    --compiler_workdir $COMPILER_DIR \
    --cache_dir $CACHE_DIR

# =============================================================================
# Step 4: Compile transformer (CausalWanModel, TP=4)
# =============================================================================
echo ""
echo "============================================================"
echo "Step 4: Compiling transformer (TP=4)..."
echo "============================================================"
WORLD_SIZE=8 neuron_parallel_compile python neuron_rolling_forcing/compile_transformer.py \
    --tp_degree 4 \
    --world_size 8 \
    --max_frames 21 \
    --height 480 \
    --width 832 \
    --checkpoint_path $CHECKPOINT_PATH \
    --compiled_models_dir $COMPILED_DIR \
    --compiler_workdir $COMPILER_DIR

echo ""
echo "============================================================"
echo "Compilation Complete!"
echo "============================================================"
echo "Compiled models saved to: $COMPILED_DIR/"
echo ""
echo "To run inference:"
echo "  python run_rolling_forcing.py --prompt 'your prompt' --output_path output.mp4"
echo "============================================================"
