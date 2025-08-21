import os
import copy
import diffusers
import math
import numpy as npy
import time
import torch
import torch_neuronx
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from torch import nn

# Define datatype
DTYPE = torch.bfloat16

import torch
from torch import nn
from transformers.models.umt5 import UMT5EncoderModel
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.autoencoders.vae import Decoder

import math

def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
  orig_shape = None
  if len(query.shape) == 4:
    orig_shape = query.shape
    def to3d(x):
      return x.reshape(-1, x.shape[2], x.shape[3])
    query, key, value = map(to3d, [query, key, value])
  if query.size() == key.size():
    attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
      1 / math.sqrt(query.size(-1))
    )
    attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
  else:
    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
      1 / math.sqrt(query.size(-1))
    )
    attention_probs = attention_scores.softmax(dim=-1)
  attn_out = torch.bmm(attention_probs, value)
  if orig_shape:
    attn_out = attn_out.reshape(
      orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
    )
  return attn_out

class TracingUMT5Wrapper(nn.Module):
  def __init__(self, t: UMT5EncoderModel, seqlen: int):
    super().__init__()
    self.t = t
    self.device = t.device
    for block_idx in range(len(self.t.encoder.block)):
      precomputed_bias = self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
      self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias
  def forward(self, text_input_ids, prompt_attention_mask):
    return self.t(
      text_input_ids,
      attention_mask=prompt_attention_mask
    )

class InferenceTextEncoderWrapper(nn.Module):
  def __init__(self, dtype, t: UMT5EncoderModel, seqlen: int):
    super().__init__()
    self.dtype = dtype
    self.device = t.device
    self.t = t
  def forward(self, text_input_ids, attention_mask=None):
    return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]

class TracingTransformerWrapper(nn.Module):
  def __init__(self, transformer):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states=None, timestep=None, encoder_hidden_states=None, **kwargs):
    return self.transformer(
      hidden_states=hidden_states,
      timestep=timestep,
      encoder_hidden_states=encoder_hidden_states,
      return_dict=False)

class InferenceTransformerWrapper(nn.Module):
  def __init__(self, transformer: WanTransformer3DModel):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states, timestep=None, encoder_hidden_states=None,
              return_dict=False):
    output = self.transformer(
      hidden_states,
      timestep,
      encoder_hidden_states)
    return output

class SimpleWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, x):
    output = self.model(x)
    return output

class f32Wrapper(nn.Module):
  def __init__(self, original):
    super().__init__()
    self.original = original
  def forward(self, x):
    t = x.dtype
    y = x.to(torch.float32)
    output = self.original(y)
    return output.type(t)

sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
  if attn_mask is not None:
    return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
  else:
    return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

torch.nn.functional.scaled_dot_product_attention = attention_wrapper

def get_pipe(dtype):
  model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="hf_pretrained_wan2.1_t2v/wan2.1_t2v_hf_cache_dir")
  pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype, cache_dir="hf_pretrained_wan2.1_t2v/wan2.1_t2v_hf_cache_dir")
  return pipe

# --- Load all compiled models ---
COMPILER_WORKDIR_ROOT = 'wan2.1_t2v_compile_dir'
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

seqlen = 300

pipe = get_pipe(DTYPE)

_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder
assert pipe._execution_device is not None

device_ids = [0, 1]  # [0, 1, 2, 3] for trn2, [0, 1] for inf2
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)
pipe.transformer = _neuronTransformer

# pipe.vae.decoder = SimpleWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load(post_quant_conv_filename))


# Run pipeline
prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

start = time.time()
output_warmup = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=256,  # default: 480
    width=256,  # default: 832
    num_frames=13,  # default: 81
    guidance_scale=5.0,
    max_sequence_length=seqlen  # default: 512
).frames[0]
end = time.time()
print('warmup time:', end-start)

start = time.time()
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=256,  # default: 480
    width=256,  # default: 832
    num_frames=13,  # default: 81
    guidance_scale=5.0,
    num_inference_steps=50,  # default: 50
    max_sequence_length=seqlen  # default: 512
).frames[0]
end = time.time()
print('time:', end-start)
export_to_video(output, "output.mp4", fps=15)

