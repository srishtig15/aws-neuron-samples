#!/bin/bash
set -e

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b
export PYTHONPATH=$(pwd):$PYTHONPATH

# Fix nearest-exact -> nearest for Trainium2 compatibility
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
else
    echo "autoencoder_kl_wan.py already patched"
fi

COMPILED_MODELS_DIR=/opt/dlami/nvme/compiled_models_t2v_a14b
COMPILER_WORKDIR=/opt/dlami/nvme/compiler_workdir_t2v_a14b/decoder_rolling
CACHE_DIR=/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir

echo "============================================"
echo "Compiling VAE Decoder with Rolling feat_cache"
echo "============================================"

WORLD_SIZE=8 neuron_parallel_compile python neuron_wan2_2_t2v_a14b/compile_decoder_rolling.py     --compiled_models_dir ${COMPILED_MODELS_DIR}     --compiler_workdir ${COMPILER_WORKDIR}     --height 480 --width 832 --num_frames 81     --decoder_frames 2     --tp_degree 8 --world_size 8     --cache_dir ${CACHE_DIR}

echo "============================================"
echo "Done! Now run inference:"
echo "  python run_wan2.2_t2v_a14b.py --compiled_models_dir ${COMPILED_MODELS_DIR} --prompt 'A cat walks on the grass, realistic' --output output_t2v_rolling.mp4"
echo "============================================"
