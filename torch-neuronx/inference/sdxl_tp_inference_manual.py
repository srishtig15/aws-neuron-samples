"""
SDXL inference with TP=4 UNet - manual diffusion loop
This avoids diffusers pipeline conflicts with Neuron TP setup
"""
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import neuronx_distributed
from diffusers import StableDiffusionXLPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import time
import numpy as np
from PIL import Image

COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16_tp'
TP_DEGREE = 4

class NeuronUNetTPWrapper(torch.nn.Module):
    def __init__(self, unet_neuron, original_unet):
        super().__init__()
        self.unet_neuron = unet_neuron
        self.config = original_unet.config
        self.in_channels = original_unet.config.in_channels
        self.device = torch.device('cpu')
        self.dtype = torch.bfloat16
        self.add_embedding = original_unet.add_embedding

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        timestep_f32 = timestep.float().expand(sample.shape[0])
        text_embeds = added_cond_kwargs['text_embeds'].to(torch.bfloat16)
        time_ids = added_cond_kwargs['time_ids'].to(torch.bfloat16)

        out = self.unet_neuron(
            sample.to(torch.bfloat16),
            timestep_f32,
            encoder_hidden_states.to(torch.bfloat16),
            text_embeds,
            time_ids
        )
        sample_out = out[0] if isinstance(out, tuple) else out
        return UNet2DConditionOutput(sample=sample_out)


def main():
    print("=== SDXL TP=4 Inference ===")

    # Load pipeline (this loads all components on CPU)
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    original_unet = pipe.unet

    # Load TP UNet
    print("Loading TP UNet...")
    unet_dir = os.path.join(COMPILER_WORKDIR_ROOT, 'unet')
    unet_neuron = neuronx_distributed.trace.parallel_model_load(unet_dir)

    # Replace UNet with TP version
    pipe.unet = NeuronUNetTPWrapper(unet_neuron, original_unet)

    # Warmup
    print("Warming up...")
    prompt = "warmup"
    with torch.no_grad():
        # Encode prompt on CPU
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

    # Run inference
    print("\n=== Running inference ===")
    prompts = [
        "A majestic mountain landscape at sunset, photorealistic",
        "A futuristic city with flying cars, digital art",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")

        with torch.no_grad():
            start_time = time.time()

            # Use the pipeline's __call__ with guidance_scale for CFG
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=1024,
                width=1024,
            ).images[0]

            elapsed = time.time() - start_time

        # Save and analyze result
        img_arr = np.array(image)
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Image shape: {img_arr.shape}, min={img_arr.min()}, max={img_arr.max()}, mean={img_arr.mean():.1f}")

        if img_arr.max() > 0:
            filename = f"sdxl_tp4_{i}.png"
            image.save(filename)
            print(f"  Saved to {filename}")
        else:
            print("  WARNING: Image appears to be black!")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
