# SDXL Inference with Tensor Parallelism on TRN2
# This version uses Tensor Parallel to reduce latency for single requests

import os
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

import time
import copy

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
import math
import torch.nn.functional as F
from typing import Optional

# For Tensor Parallelism
import neuronx_distributed
from neuronx_distributed.trace import parallel_model_trace
from neuronx_distributed.parallel_layers import parallel_state

# ============== Attention Kernels ==============
_flash_fwd_call = nki_jit()(attention_isa_kernel)

def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))
    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    scale = 1 / math.sqrt(d_head)
    _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


class KernelizedAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored."
            diffusers.utils.deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None or query.shape[3] > query.shape[2] or query.shape[3] > 128 or value.shape[2] == 77:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = attention_wrapper_without_swap(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


# ============== Model Wrappers ==============
class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
        diffusers.models.attention_processor.AttnProcessor2_0.__call__ = KernelizedAttnProcessor2_0.__call__

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"]
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(text_embeds=out_tuple[0], last_hidden_state=out_tuple[1], hidden_states=out_tuple[2])


class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        out_tuple = self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)
        return out_tuple


# ============== Configuration ==============
COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16_tp'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Tensor Parallel degree - use 2 cores for UNet
# This splits the model across 2 cores to reduce latency
TP_DEGREE = 8


# ============== Compilation Functions ==============
def compile_text_encoders():
    """Compile text encoders (no TP needed, they're small)"""
    print("=== Compiling Text Encoders ===")

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
    traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))
    del pipe

    text_input_ids_1 = torch.tensor([[49406, 736, 1615, 49407] + [49407] * 73])
    text_input_ids_2 = torch.tensor([[49406, 736, 1615, 49407] + [0] * 73])

    # Text Encoder 1
    print("Compiling Text Encoder 1...")
    neuron_text_encoder = torch_neuronx.trace(
        traceable_text_encoder,
        text_input_ids_1,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
        compiler_args=["--lnc=2"],
    )
    torch.jit.save(neuron_text_encoder, os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt'))

    # Text Encoder 2
    print("Compiling Text Encoder 2...")
    neuron_text_encoder_2 = torch_neuronx.trace(
        traceable_text_encoder_2,
        text_input_ids_2,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
        compiler_args=["--lnc=2"],
    )
    torch.jit.save(neuron_text_encoder_2, os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt'))
    print("Text Encoders compiled successfully!")


def compile_vae():
    """Compile VAE decoder and post_quant_conv (no TP needed)"""
    print("=== Compiling VAE ===")

    # VAE Decoder
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    print("Compiling VAE Decoder...")
    decoder_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
    decoder_neuron = torch_neuronx.trace(
        decoder,
        decoder_in,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
        compiler_args=["--model-type=unet-inference", "--lnc=2"],
    )
    torch.jit.save(decoder_neuron, os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt'))
    del decoder

    # VAE post_quant_conv
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe

    print("Compiling VAE post_quant_conv...")
    post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
    post_quant_conv_neuron = torch_neuronx.trace(
        post_quant_conv,
        post_quant_conv_in,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
        compiler_args=["--lnc=2"],
    )
    torch.jit.save(post_quant_conv_neuron, os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt'))
    print("VAE compiled successfully!")


def get_unet_model():
    """Factory function to create UNet model for TP tracing

    Returns (model, input_output_alias) tuple as required by parallel_model_trace
    """
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    Attention.get_attention_scores = get_attention_scores_neuron
    unet = UNetWrap(pipe.unet)
    # Return tuple of (model, input_output_alias)
    # input_output_alias is None since we don't need aliasing
    return unet, None


def compile_unet_tp():
    """Compile UNet with Tensor Parallelism"""
    print(f"=== Compiling UNet with TP={TP_DEGREE} ===")

    # Example inputs for tracing - BF16
    sample_1b = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 2048], dtype=torch.bfloat16)
    text_embeds = torch.randn([1, 1280], dtype=torch.bfloat16)
    time_ids = torch.randn([1, 6], dtype=torch.bfloat16)
    example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, text_embeds, time_ids)

    print(f"Compiling UNet with Tensor Parallel degree={TP_DEGREE}...")
    print("This may take 1-2 hours...")

    os.makedirs(os.path.join(COMPILER_WORKDIR_ROOT, 'unet'), exist_ok=True)

    # Use parallel_model_trace for Tensor Parallelism
    # Pass the factory function, not the model instance
    unet_neuron = parallel_model_trace(
        get_unet_model,
        example_inputs,
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
        compiler_args="--model-type=unet-inference --lnc=2",
    )

    # Save the TP model
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    torch.jit.save(unet_neuron, unet_filename)
    print("UNet with TP compiled successfully!")


def compile_all():
    """Compile all models"""
    os.makedirs(COMPILER_WORKDIR_ROOT, exist_ok=True)
    compile_text_encoders()
    compile_vae()
    compile_unet_tp()
    print("\n=== All models compiled successfully! ===")


# ============== Inference ==============
def run_inference():
    """Run inference with TP UNet"""
    print("=== Loading compiled models ===")

    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Load UNet with TP - no DataParallel needed
    # The model is already split across TP_DEGREE cores
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    pipe.unet.unetwrap = torch.jit.load(unet_filename)

    # Load other models
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load(text_encoder_filename), pipe.text_encoder)
    pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load(text_encoder_2_filename), pipe.text_encoder_2)

    print("=== Running warmup ===")
    prompt = "a photo of an astronaut riding a horse on mars"
    image_warmup = pipe(prompt).images[0]

    print("=== Running inference ===")
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "sonic on the moon",
        "elvis playing guitar while eating a hotdog",
        "A digital illustration of a steampunk flying machine",
    ]

    total_time = 0
    for i, p in enumerate(prompts):
        start_time = time.time()
        image = pipe(p).images[0]
        elapsed = time.time() - start_time
        total_time += elapsed
        image.save(f"image_tp_{i}.png")
        print(f"Prompt {i+1}: {elapsed:.2f} seconds")

    print(f"\nAverage time per image: {total_time/len(prompts):.2f} seconds")
    print(f"With Tensor Parallel (TP={TP_DEGREE}), latency should be lower than DataParallel")


# ============== Main ==============
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        compile_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "compile_unet":
        compile_unet_tp()
    elif len(sys.argv) > 1 and sys.argv[1] == "inference":
        run_inference()
    else:
        print("Usage:")
        print("  python hf_pretrained_sdxl_1024_tp.py compile      # Compile all models")
        print("  python hf_pretrained_sdxl_1024_tp.py compile_unet # Compile only UNet with TP")
        print("  python hf_pretrained_sdxl_1024_tp.py inference    # Run inference")
        print("")
        print("Note: Compilation takes 1-2 hours for UNet")
