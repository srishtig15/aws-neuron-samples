import os
os.environ['NEURON_FUSE_SOFTMAX'] = '1'
os.environ['NEURON_RT_VIRTUAL_CORE_SIZE'] = '2'
os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
import neuronx_distributed
from neuronx_distributed.parallel_layers import parallel_state
import time
import numpy as np

COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16_tp'
TP_DEGREE = 8

class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(last_hidden_state=out_tuple[0], text_embeds=out_tuple[1], hidden_states=out_tuple[2])

class NeuronUNetTPWrapper(nn.Module):
    def __init__(self, unet_neuron, original_unet):
        super().__init__()
        self.unet_neuron = unet_neuron
        self.config = original_unet.config
        self.in_channels = original_unet.config.in_channels
        self.add_embedding = original_unet.add_embedding
        self.device = torch.device('cpu')
        self.dtype = torch.bfloat16

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        timestep_bf16 = timestep.to(torch.bfloat16).expand(sample.shape[0])
        text_embeds = added_cond_kwargs['text_embeds'].to(torch.bfloat16)
        time_ids = added_cond_kwargs['time_ids'].to(torch.bfloat16)

        out = self.unet_neuron(
            sample.to(torch.bfloat16),
            timestep_bf16,
            encoder_hidden_states.to(torch.bfloat16),
            text_embeds,
            time_ids
        )
        sample_out = out[0] if isinstance(out, tuple) else out
        return UNet2DConditionOutput(sample=sample_out)

def main():
    # Initialize distributed
    dist.init_process_group(backend='gloo')

    # Initialize model parallel
    neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP_DEGREE
    )

    rank = parallel_state.get_tensor_model_parallel_rank()

    if rank == 0:
        print('=== Loading Pipeline ===')
    pipe = DiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    original_unet = pipe.unet

    if rank == 0:
        print('Loading TP UNet...')
    unet_tp = neuronx_distributed.trace.parallel_model_load(
        f'{COMPILER_WORKDIR_ROOT}/unet'
    )
    pipe.unet = NeuronUNetTPWrapper(unet_tp, original_unet)

    if rank == 0:
        print('Loading text encoders...')
    text_encoder_neuron = torch.jit.load(f'{COMPILER_WORKDIR_ROOT}/text_encoder/model.pt')
    text_encoder_2_neuron = torch.jit.load(f'{COMPILER_WORKDIR_ROOT}/text_encoder_2/model.pt')
    pipe.text_encoder = TextEncoderOutputWrapper(text_encoder_neuron, pipe.text_encoder)
    pipe.text_encoder_2 = TextEncoderOutputWrapper(text_encoder_2_neuron, pipe.text_encoder_2)

    if rank == 0:
        print('Loading VAE...')
    vae_decoder_neuron = torch.jit.load(f'{COMPILER_WORKDIR_ROOT}/vae_decoder/model.pt')
    post_quant_conv_neuron = torch.jit.load(f'{COMPILER_WORKDIR_ROOT}/vae_post_quant_conv/model.pt')
    pipe.vae.decoder = vae_decoder_neuron
    pipe.vae.post_quant_conv = post_quant_conv_neuron

    if rank == 0:
        print('\n=== Running Warmup ===')
    warmup_prompt = 'warmup'
    _ = pipe(warmup_prompt, num_inference_steps=30, height=1024, width=1024)

    if rank == 0:
        print('\n=== Running Inference ===')
    prompt = 'A beautiful sunset over mountains, photorealistic, 8k'
    start = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024
    ).images[0]
    elapsed = time.time() - start

    if rank == 0:
        img_arr = np.array(image)
        print(f'Image shape: {img_arr.shape}')
        print(f'Image min: {img_arr.min()}, max: {img_arr.max()}, mean: {img_arr.mean():.2f}')
        print(f'Non-zero pixels: {np.count_nonzero(img_arr)} / {img_arr.size}')

        if img_arr.max() > 0:
            image.save('sdxl_tp_test.png')
            print(f'Saved to sdxl_tp_test.png')
            print(f'\nTotal time: {elapsed:.2f}s for 30 steps')
            print(f'UNet throughput: {30/elapsed:.2f} it/s')
        else:
            print('WARNING: Image is still black!')

if __name__ == '__main__':
    main()
