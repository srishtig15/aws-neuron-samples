# The script can run on Trn2
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

# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import time
import copy
# from IPython.display import clear_output

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402
import math
import torch.nn.functional as F
from typing import Optional

_flash_fwd_call = nki_jit()(attention_isa_kernel)
def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape  # my change
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
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

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
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
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
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
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

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if attention_mask is not None or query.shape[3] > query.shape[2] or query.shape[3] > 128 or value.shape[2] == 77:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = attention_wrapper_without_swap(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# clear_output(wait=False)


def get_attention_scores_neuron(self, query, key, attn_mask):    
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs
 

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled
 

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
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
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
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
    

# COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16'

# # Model ID for SD XL version pipeline
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# # --- Compile Text Encoders and save --- (using BF16)

# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)


# # Apply wrappers to make text encoders traceable
# traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
# traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))

# del pipe

# text_input_ids_1 = torch.tensor([[49406,   736,  1615, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407]])


# text_input_ids_2 = torch.tensor([[49406,   736,  1615, 49407,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]])


# # Text Encoder 1
# neuron_text_encoder = torch_neuronx.trace(
#     traceable_text_encoder,
#     text_input_ids_1,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
#     compiler_args=["--lnc=2"],
# )

# text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
# torch.jit.save(neuron_text_encoder, text_encoder_filename)


# # Text Encoder 2
# neuron_text_encoder_2 = torch_neuronx.trace(
#     traceable_text_encoder_2,
#     text_input_ids_2,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
#     compiler_args=["--lnc=2"],
# )

# text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
# torch.jit.save(neuron_text_encoder_2, text_encoder_2_filename)

# # --- Compile VAE decoder and save ---

# # Only keep the model being compiled in RAM to minimze memory pressure
# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
# decoder = copy.deepcopy(pipe.vae.decoder)
# del pipe

# # Compile vae decoder - BF16
# decoder_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
# decoder_neuron = torch_neuronx.trace(
#     decoder,
#     decoder_in,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
#     compiler_args=["--model-type=unet-inference", "--lnc=2"],
# )

# # Save the compiled vae decoder
# decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
# torch.jit.save(decoder_neuron, decoder_filename)

# # delete unused objects
# del decoder


# # --- Compile UNet and save ---

# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

# # Replace original cross-attention module with custom cross-attention module for better performance
# Attention.get_attention_scores = get_attention_scores_neuron

# # Apply double wrapper to deal with custom return type
# pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

# # Only keep the model being compiled in RAM to minimze memory pressure
# unet = copy.deepcopy(pipe.unet.unetwrap)
# del pipe

# # Compile unet - BF16
# sample_1b = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
# timestep_1b = torch.tensor(999).float().expand((1,))
# encoder_hidden_states_1b = torch.randn([1, 77, 2048], dtype=torch.bfloat16)
# added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280], dtype=torch.bfloat16),
#                         "time_ids": torch.randn([1, 6], dtype=torch.bfloat16)}
# example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

# unet_neuron = torch_neuronx.trace(
#     unet,
#     example_inputs,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
#     compiler_args=["--model-type=unet-inference", "--lnc=2"]
# )

# # save compiled unet
# unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
# torch.jit.save(unet_neuron, unet_filename)

# # delete unused objects
# del unet


# # --- Compile VAE post_quant_conv and save ---

# # Only keep the model being compiled in RAM to minimze memory pressure
# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
# post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
# del pipe

# # Compile vae post_quant_conv - BF16
# post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
# post_quant_conv_neuron = torch_neuronx.trace(
#     post_quant_conv,
#     post_quant_conv_in,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
#     compiler_args=["--lnc=2"],
# )

# # Save the compiled vae post_quant_conv
# post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
# torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# # delete unused objects
# del post_quant_conv


# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = 'sdxl_compile_dir_1024_bf16'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Load the compiled UNet onto neuron cores.
# With BF16, each model copy is ~5GB, so we can use all 8 cores with LNC=2
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
# device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 cores
device_ids = [0, 1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load(text_encoder_filename), pipe.text_encoder)
pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load(text_encoder_2_filename), pipe.text_encoder_2)

# Run pipeline
prompt = ["a photo of an astronaut riding a horse on mars",
          "sonic on the moon",
          "elvis playing guitar while eating a hotdog",
          "saved by the bell",
          "engineers eating lunch at the opera",
          "panda eating bamboo on a plane",
          "A digital illustration of a steampunk flying machine in the sky with cogs and mechanisms, 4k, detailed, trending in artstation, fantasy vivid colors",
          "kids playing soccer at the FIFA World Cup"
         ]

# First do a warmup run so all the asynchronous loads can finish
image_warmup = pipe(prompt[0]).images[0]

# plt.title("Image")
# plt.xlabel("X pixel scaling")
# plt.ylabel("Y pixels scaling")

total_time = 0
for x in prompt:
    start_time = time.time()
    image = pipe(x).images[0]
    total_time = total_time + (time.time()-start_time)
    image.save("image.png")
    # image = mpimg.imread("image.png")
    #clear_output(wait=True)
    # plt.imshow(image)
    # plt.show()
print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")


prompt = "A cat holding a sign that says hello world"

start_time = time.time()
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
end_time = time.time()
generation_time = end_time - start_time
image.save("flux-dev.png")
print(f"Image generated in {generation_time:.2f} seconds")