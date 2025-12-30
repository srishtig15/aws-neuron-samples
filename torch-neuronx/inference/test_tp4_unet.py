"""Simple test to verify TP=4 UNet can load and run"""
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import neuronx_distributed

COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16_tp'

def main():
    print("=== Loading TP UNet ===")
    unet_dir = os.path.join(COMPILER_WORKDIR_ROOT, 'unet')

    print(f"Loading from {unet_dir}")
    unet_neuron = neuronx_distributed.trace.parallel_model_load(unet_dir)
    print(f"Model loaded successfully: {type(unet_neuron)}")

    # Test with example inputs matching compilation
    print("\n=== Creating test inputs ===")
    batch_size = 2
    # Match dtypes used during compilation
    sample = torch.randn([batch_size, 4, 128, 128], dtype=torch.bfloat16)
    timestep = torch.tensor(999).float().expand((batch_size,))  # float32 like compilation
    encoder_hidden_states = torch.randn([batch_size, 77, 2048], dtype=torch.bfloat16)
    text_embeds = torch.randn([batch_size, 1280], dtype=torch.bfloat16)
    time_ids = torch.randn([batch_size, 6], dtype=torch.bfloat16)

    print(f"sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"timestep shape: {timestep.shape}, dtype: {timestep.dtype}")
    print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}, dtype: {encoder_hidden_states.dtype}")
    print(f"text_embeds shape: {text_embeds.shape}, dtype: {text_embeds.dtype}")
    print(f"time_ids shape: {time_ids.shape}, dtype: {time_ids.dtype}")

    print("\n=== Running forward pass (10 iterations) ===")
    import time
    for i in range(10):
        with torch.no_grad():
            start = time.time()
            output = unet_neuron(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
            elapsed = time.time() - start

        sample_out = output[0] if isinstance(output, tuple) else output
        print(f"Iteration {i+1}: {elapsed:.3f}s, output min/max: {sample_out.min():.4f} / {sample_out.max():.4f}")

    print("\n=== SUCCESS - All 10 iterations completed ===")

if __name__ == "__main__":
    main()
