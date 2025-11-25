#!/bin/bash

# source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
# cp autoencoder_kl_wan.py /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/diffusers/models/autoencoders/  # Trainium2 doesn't support 'nearest-exact'

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
# --num_frames 61

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
# --num_frames 61

python neuron_wan2_2_ti2v/compile_decoder_tp.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--height 512 \
--width 512

echo "compiling encoder"
python neuron_wan2_2_ti2v/compile_encoder.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--height 512 \
--width 512

# python neuron_wan2_2_ti2v/compile_encoder.py \
# --compiled_models_dir "compile_workdir_latency_optimized" \
# --height 720 \
# --width 1280

echo "run wan2.2 ti2v latency optimized"
export NEURON_RT_NUM_CORES=4
python run_wan2.2_ti2v_latency_optimized.py
python run_wan2.2_i2v_latency_optimized.py
