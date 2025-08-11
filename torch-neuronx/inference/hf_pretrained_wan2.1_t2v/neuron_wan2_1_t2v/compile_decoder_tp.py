import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
# os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
# os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
# compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import copy
import torch
import argparse
import torch_neuronx
import neuronx_distributed
from diffusers import AutoencoderKLWan, WanPipeline
from torch import nn
from functools import partial

# 导入分片工具函数
from neuron_parallel_utils import shard_conv2d, shard_conv3d, shard_linear, shard_transformer_attn

class TracingVAEDecoderWrapper(nn.Module):
    def __init__(self, vae_decoder):
        super().__init__()
        self.decoder = vae_decoder
        self.config = vae_decoder.config if hasattr(vae_decoder, 'config') else None
        self.dtype = next(vae_decoder.parameters()).dtype
        self.device = next(vae_decoder.parameters()).device
    
    def forward(self, latents):
        return self.decoder.decode(latents, return_dict=False)[0]

def shard_resnet_block(resnet_block, tp_degree):
    """分片 ResNet block"""
    if hasattr(resnet_block, 'conv1'):
        resnet_block.conv1 = shard_conv3d(resnet_block.conv1, tp_degree, dim=0)
    
    if hasattr(resnet_block, 'conv2'):
        resnet_block.conv2 = shard_conv3d(resnet_block.conv2, tp_degree, dim=1)
    
    if hasattr(resnet_block, 'conv_shortcut') and resnet_block.conv_shortcut is not None:
        resnet_block.conv_shortcut = shard_conv3d(resnet_block.conv_shortcut, tp_degree, dim=0)
    
    return resnet_block

def shard_attention_block(attn_block, tp_degree):
    """分片 Attention block"""
    if hasattr(attn_block, 'to_q'):
        attn_block.to_q = shard_linear(attn_block.to_q, tp_degree, dim=0)
    
    if hasattr(attn_block, 'to_k'):
        attn_block.to_k = shard_linear(attn_block.to_k, tp_degree, dim=0)
    
    if hasattr(attn_block, 'to_v'):
        attn_block.to_v = shard_linear(attn_block.to_v, tp_degree, dim=0)
    
    if hasattr(attn_block, 'to_out') and len(attn_block.to_out) > 0:
        attn_block.to_out[0] = shard_linear(attn_block.to_out[0], tp_degree, dim=1)
    
    return attn_block

def shard_vae_decoder_layers(decoder, tp_degree):
    """
    对 VAE decoder 的层进行 tensor 并行分片
    """
    print(f"Sharding VAE decoder with tp_degree={tp_degree}")
    
    # 分片输入卷积层
    if hasattr(decoder, 'conv_in'):
        print("Sharding conv_in")
        decoder.conv_in = shard_conv3d(decoder.conv_in, tp_degree, dim=0)
    
    # 分片 mid_block
    if hasattr(decoder, 'mid_block'):
        print("Sharding mid_block")
        mid_block = decoder.mid_block
        
        # 分片 ResNet blocks
        if hasattr(mid_block, 'resnets'):
            for i, resnet in enumerate(mid_block.resnets):
                print(f"  Sharding mid_block resnet {i}")
                mid_block.resnets[i] = shard_resnet_block(resnet, tp_degree)
        
        # 分片 attention blocks
        if hasattr(mid_block, 'attentions'):
            for i, attn in enumerate(mid_block.attentions):
                if attn is not None:
                    print(f"  Sharding mid_block attention {i}")
                    mid_block.attentions[i] = shard_attention_block(attn, tp_degree)
    
    # 分片 up_blocks
    if hasattr(decoder, 'up_blocks'):
        for block_idx, up_block in enumerate(decoder.up_blocks):
            print(f"Sharding up_block {block_idx}")
            
            # 分片 ResNet blocks
            if hasattr(up_block, 'resnets'):
                for resnet_idx, resnet in enumerate(up_block.resnets):
                    print(f"  Sharding up_block {block_idx} resnet {resnet_idx}")
                    up_block.resnets[resnet_idx] = shard_resnet_block(resnet, tp_degree)
            
            # 分片 attention blocks (如果存在)
            if hasattr(up_block, 'attentions'):
                for attn_idx, attn in enumerate(up_block.attentions):
                    if attn is not None:
                        print(f"  Sharding up_block {block_idx} attention {attn_idx}")
                        up_block.attentions[attn_idx] = shard_attention_block(attn, tp_degree)
            
            # 分片 upsample layers
            if hasattr(up_block, 'upsamplers') and up_block.upsamplers:
                for upsampler_idx, upsampler in enumerate(up_block.upsamplers):
                    if hasattr(upsampler, 'conv'):
                        print(f"  Sharding up_block {block_idx} upsampler {upsampler_idx}")
                        upsampler.conv = shard_conv3d(upsampler.conv, tp_degree, dim=0)
    
    # 分片输出层
    if hasattr(decoder, 'conv_out'):
        print("Sharding conv_out")
        # 输出卷积层需要沿输入通道分片以匹配前面的分片
        decoder.conv_out = shard_conv3d(decoder.conv_out, tp_degree, dim=1)
    
    return decoder

def get_vae_decoder(tp_degree: int):
    """
    获取并分片 VAE decoder
    """
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DTYPE = torch.float32  # VAE 通常使用 float32
    
    # 加载 VAE
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=DTYPE, 
        cache_dir="wan2.1_t2v_hf_cache_dir"
    )
    vae.eval()
    
    # 分片 decoder
    if tp_degree > 1:
        vae.decoder = shard_vae_decoder_layers(vae.decoder, tp_degree)
    
    # 创建包装器
    decoder_wrapper = TracingVAEDecoderWrapper(vae)
    
    return decoder_wrapper, {}

def compile_vae_decoder(args):
    """
    编译 VAE decoder
    """
    # 设置 tensor 并行参数
    tp_degree = 8  # 使用 tp=8
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    
    # 输入参数
    batch_size = 1
    in_channels = 16  # VAE latent channels
    frames = args.frames
    latent_height = args.height // 8  # VAE 下采样因子为 8
    latent_width = args.width // 8
    
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    # 创建获取模型的函数
    get_vae_decoder_f = partial(get_vae_decoder, tp_degree)
    
    with torch.no_grad():
        # 创建示例输入 - VAE latents
        sample_latents = torch.randn(
            (batch_size, in_channels, frames, latent_height, latent_width), 
            dtype=torch.float32
        )
        
        print(f"Sample latents shape: {sample_latents.shape}")
        print(f"Compiling VAE decoder with tp_degree={tp_degree}")
        
        # 编译模型
        compiled_vae_decoder = neuronx_distributed.trace.parallel_model_trace(
            get_vae_decoder_f,
            sample_latents,
            compiler_workdir=f"{compiler_workdir}/vae_decoder",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )
        
        # 保存编译后的模型
        compiled_model_dir = f"{compiled_models_dir}/vae_decoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
            
        neuronx_distributed.trace.parallel_model_save(
            compiled_vae_decoder, f"{compiled_model_dir}"
        )
        
        print(f"VAE decoder compiled successfully with tp_degree={tp_degree}")
        print(f"Saved to: {compiled_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video", type=int, default=256)
    parser.add_argument("--width", help="width of generated video", type=int, default=256)
    parser.add_argument("--frames", help="number of frames", type=int, default=21)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_vae_decoder(args)
