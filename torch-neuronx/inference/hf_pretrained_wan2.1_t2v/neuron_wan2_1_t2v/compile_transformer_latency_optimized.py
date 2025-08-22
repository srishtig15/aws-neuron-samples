import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --internal-hlo2tensorizer-options='--fuse-dot-logistic=false' --model-type=transformer --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=transformer --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan, WanPipeline
import torch
import copy
import argparse
import neuronx_distributed
import torch_neuronx

from torch import nn
import torch.nn.functional as F
from functools import partial

from neuron_commons import attention_wrapper, attention_wrapper_for_transformer, f32Wrapper
from neuron_parallel_utils import shard_transformer_attn, shard_transformer_feedforward, shard_transformer3d_attn

# torch.nn.functional.scaled_dot_product_attention = attention_wrapper  # TODO use attention_wrapper instead of attention_wrapper_for_transformer
# torch.nn.functional.scaled_dot_product_attention = attention_wrapper_for_transformer


# def upcast_norms_to_f32(transformer):
#     transformer.condition_embedder.time_embedder = f32Wrapper(transformer.condition_embedder.time_embedder)
#     for block in transformer.blocks:
#         orig_norm1 = block.norm1
#         orig_norm2 = block.norm2
#         orig_norm3 = block.norm3
#         block.norm1 = f32Wrapper(orig_norm1)
#         block.norm2 = f32Wrapper(orig_norm2)
#         block.norm3 = f32Wrapper(orig_norm3)
#     orig_norm_out = transformer.norm_out
#     transformer.norm_out = f32Wrapper(orig_norm_out)
    
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

def get_transformer_model(tp_degree: int):
    DTYPE = torch.bfloat16
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
    
    # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    # vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_14b_hf_cache_dir")
    # pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_14b_hf_cache_dir")
    
    # upcast_norms_to_f32(pipe.transformer)
    
    # 分片所有blocks
    for block_idx, block in enumerate(pipe.transformer.blocks):
        print(f"Processing block {block_idx+1}/{len(pipe.transformer.blocks)}")
        
        # 分片attention层
        # block.attn1 = shard_transformer_attn(tp_degree, block.attn1)
        # block.attn2 = shard_transformer_attn(tp_degree, block.attn2)
        block.attn1 = shard_transformer3d_attn(tp_degree, block.attn1)
        block.attn2 = shard_transformer3d_attn(tp_degree, block.attn2)

        # 分片feedforward层
        block.ffn = shard_transformer_feedforward(block.ffn)
        
    mod_pipe_transformer_f = TracingTransformerWrapper(pipe.transformer)
    return mod_pipe_transformer_f, {}

def compile_transformer(args):
    tp_degree = 4
    os.environ["LOCAL_WORLD_SIZE"] = "4" # Use tensor parallel degree as 4 for trn2
    # tp_degree = 8 # Use tensor parallel degree as 8 for trn1/inf2, default: 8
    # os.environ["LOCAL_WORLD_SIZE"] = "8" # Use tensor parallel degree as 4 for trn2
    latent_height = args.height//8
    latent_width = args.width//8
    num_prompts = 1
    num_images_per_prompt = args.num_images_per_prompt
    max_sequence_length = args.max_sequence_length
    hidden_size = 4096
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    batch_size = 1
    frames = 4  # default: 21
    # height, width = 32, 32  # default: 96, 96
    in_channels = 16
    sample_hidden_states = torch.ones((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.ones((batch_size, max_sequence_length, hidden_size), dtype=torch.bfloat16)
    sample_timestep = torch.ones((batch_size), dtype=torch.int64)

    get_transformer_model_f = partial(get_transformer_model, tp_degree)
    with torch.no_grad():
        sample_inputs = sample_hidden_states, sample_timestep, sample_encoder_hidden_states
        compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
            get_transformer_model_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/transformer",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )
        compiled_model_dir = f"{compiled_models_dir}/transformer"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)         
        neuronx_distributed.trace.parallel_model_save(
            compiled_transformer, f"{compiled_model_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--height", help="height of generated video.", type=int, default=256)
#     parser.add_argument("--width", help="width of generated video.", type=int, default=256)
#     parser.add_argument("--num_images_per_prompt", help="number of images per prompt.", type=int, default=1)
#     parser.add_argument("--max_sequence_length", help="max sequence length.", type=int, default=300)
#     parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
#     parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
#     args = parser.parse_args()
#     compile_transformer(args)


DTYPE=torch.bfloat16
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
COMPILER_WORKDIR_ROOT = 'compile_workdir_latency_optimized'

batch_size = 1
frames = 4  # default: 16  # typical frame count for video generation
height, width = 32, 32  # default: 96, 96  # spatial dimensions
in_channels = 16  # 根据配置，Wan使用16个输入通道
max_sequence_length = 512
hidden_size = 4096

# --- Compile Transformer and save [PASS]---

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")

# upcast_norms_to_f32(pipe.transformer)

# # Apply double wrapper to deal with custom return type
pipe.transformer = TracingTransformerWrapper(pipe.transformer)

# Only keep the model being compiled in RAM to minimze memory pressure
transformer = copy.deepcopy(pipe.transformer)

del pipe

# Compile transformer - adjust input shapes for 3D video

# 3D input for video: (batch, channels, frames, height, width)
hidden_states_1b = torch.randn([batch_size, in_channels, frames, height, width], dtype=DTYPE)
timestep_1b = torch.tensor([999], dtype=torch.int64)
# Text encoder output dimension for Wan (might be different from SD)
encoder_hidden_states_1b = torch.randn([batch_size, max_sequence_length, hidden_size], dtype=DTYPE)  # Wan uses 4096 dim

example_inputs = hidden_states_1b, timestep_1b, encoder_hidden_states_1b

transformer_neuron = torch_neuronx.trace(
    transformer,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'transformer'),
    compiler_args=compiler_flags,
    inline_weights_to_neff=False
)

# Enable asynchronous and lazy loading to speed up model load
torch_neuronx.async_load(transformer_neuron)
torch_neuronx.lazy_load(transformer_neuron)

# save compiled transformer
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
torch.jit.save(transformer_neuron, transformer_filename)

# delete unused objects
del transformer
del transformer_neuron