#!/bin/bash

# source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
# pip install -r requirements.txt
# cp autoencoder_kl_wan.py /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.10/site-packages/diffusers/models/autoencoders/  # Trainium2 doesn't support 'nearest-exact'

# Fix nearest-exact -> nearest for Trainium2 compatibility
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
fi

export PYTHONPATH=`pwd`:$PYTHONPATH

echo "cache hf model"
python neuron_wan2_2_ti2v/cache_hf_model.py

echo "compiling text encoder"
python neuron_wan2_2_ti2v/compile_text_encoder.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--max_sequence_length 512

echo "compiling transformer"
python neuron_wan2_2_ti2v/compile_transformer_latency_optimized.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--max_sequence_length 512 \
--height 512 \
--width 512 \
--num_frames 81

# python neuron_wan2_2_ti2v/compile_transformer_latency_optimized.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --max_sequence_length 512 \
# --height 720 \
# --width 1280 \
# --num_frames 81

echo "compiling decoder"
python neuron_wan2_2_ti2v/compile_decoder.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--height 512 \
--width 512 \
--num_frames 81

# python neuron_wan2_2_ti2v/compile_decoder.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --height 720 \
# --width 1280 \
# --num_frames 81

# python neuron_wan2_2_ti2v/compile_decoder_tp.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --height 720 \
# --width 1280 \
# --tp_degree 8

# echo "compiling encoder"
# python neuron_wan2_2_ti2v/compile_encoder.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --height 512 \
# --width 512

# python neuron_wan2_2_ti2v/compile_encoder.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --height 720 \
# --width 1280

echo "run wan2.2 ti2v latency optimized"
# export NEURON_RT_NUM_CORES=4
export NEURON_RT_NUM_CORES=8
python run_wan2.2_ti2v_latency_optimized.py
python run_wan2.2_i2v_latency_optimized.py
