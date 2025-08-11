import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
# os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
# os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
# compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import argparse
import neuronx_distributed
from torch import nn
from functools import partial
from diffusers.models.autoencoders.vae import Decoder
from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import shard_conv2d, shard_conv3d, shard_linear, get_sharded_data
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear

torch.nn.functional.scaled_dot_product_attention = attention_wrapper

class DecoderWrapper(nn.Module):
    """包装decoder以便于追踪"""
    def __init__(self, decoder, post_quant_conv):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
    
    def forward(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

def shard_wan_norm(norm, tp_degree):
    """分片WanNorm层的参数"""
    if hasattr(norm, 'gamma') and norm.gamma is not None:
        # 获取原始gamma的通道数
        orig_channels = norm.gamma.shape[0]
        # 计算分片后的通道数
        sharded_channels = orig_channels // tp_degree
        # 分片gamma和bias
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        start_idx = tp_rank * sharded_channels
        end_idx = start_idx + sharded_channels
        
        norm.gamma = nn.Parameter(norm.gamma[start_idx:end_idx].clone())
        if hasattr(norm, 'bias') and norm.bias is not None:
            norm.bias = nn.Parameter(norm.bias[start_idx:end_idx].clone())
    
    return norm

def shard_resnet_block(resnet, tp_degree):
    """分片ResNet块"""
    # 分片conv1 (Conv3d)
    if hasattr(resnet, 'conv1') and isinstance(resnet.conv1, nn.Conv3d):
        resnet.conv1 = shard_conv3d(resnet.conv1, tp_degree, dim=0)
        
        # 分片对应的norm2（在conv1之后）
        if hasattr(resnet, 'norm2'):
            # 检查是否是f32Wrapper
            if hasattr(resnet.norm2, 'original'):
                # 分片内部的WanNorm
                resnet.norm2.original = shard_wan_norm(resnet.norm2.original, tp_degree)
            else:
                resnet.norm2 = shard_wan_norm(resnet.norm2, tp_degree)
    
    # 分片conv2 (Conv3d)
    if hasattr(resnet, 'conv2') and isinstance(resnet.conv2, nn.Conv3d):
        # conv2输入已分片，输出需要回到原始维度（或保持分片）
        # 这里我们保持分片
        resnet.conv2 = shard_conv3d(resnet.conv2, tp_degree, dim=1)
        resnet.conv2 = shard_conv3d(resnet.conv2, 1, dim=0)  # 输出不分片
    
    # 如果有conv_shortcut且不是Identity层，则分片
    if hasattr(resnet, 'conv_shortcut') and resnet.conv_shortcut is not None:
        if isinstance(resnet.conv_shortcut, nn.Conv3d):
            # shortcut需要和主路径输出维度匹配
            resnet.conv_shortcut = shard_conv3d(resnet.conv_shortcut, 1, dim=0)  # 输出不分片
        elif isinstance(resnet.conv_shortcut, nn.Identity):
            pass
    
    return resnet

def shard_attention_block(attn, tp_degree):
    """分片注意力块 - 简化版本，避免分片"""
    # 由于VAE的attention比较小，我们可以选择不分片
    return attn

def upcast_norms_to_f32(decoder: Decoder):
    """将归一化层转换为FP32"""
    if hasattr(decoder, 'up_blocks'):
        for upblock in decoder.up_blocks:
            if hasattr(upblock, 'resnets'):
                for resnet in upblock.resnets:
                    if hasattr(resnet, 'norm1'):
                        orig_resnet_norm1 = resnet.norm1
                        resnet.norm1 = f32Wrapper(orig_resnet_norm1)
                    if hasattr(resnet, 'norm2'):
                        orig_resnet_norm2 = resnet.norm2
                        resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    
    if hasattr(decoder, 'mid_block') and decoder.mid_block is not None:
        if hasattr(decoder.mid_block, 'resnets'):
            for resnet in decoder.mid_block.resnets:
                if hasattr(resnet, 'norm1'):
                    orig_resnet_norm1 = resnet.norm1
                    resnet.norm1 = f32Wrapper(orig_resnet_norm1)
                if hasattr(resnet, 'norm2'):
                    orig_resnet_norm2 = resnet.norm2
                    resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    
    if hasattr(decoder, 'norm_out'):
        orig_norm_out = decoder.norm_out
        decoder.norm_out = f32Wrapper(orig_norm_out)

class ShardedConvIn(nn.Module):
    """处理conv_in的特殊包装器，广播输入到所有分片"""
    def __init__(self, conv_layer, tp_degree):
        super().__init__()
        self.conv = conv_layer
        self.tp_degree = tp_degree
        
    def forward(self, x):
        # 对输入进行处理，确保每个分片都获得完整输入
        return self.conv(x)

def shard_decoder(decoder: Decoder, tp_degree: int):
    """对decoder进行简化的分片策略"""
    
    # conv_in保持原样，不分片
    # 它接收16通道输入，输出384通道
    
    # 只分片mid_block的ResNet块，不分片attention
    if hasattr(decoder, 'mid_block') and decoder.mid_block is not None:
        if hasattr(decoder.mid_block, 'resnets'):
            for resnet in decoder.mid_block.resnets:
                shard_resnet_block(resnet, tp_degree)
        
        # 不分片attention层，保持简单
        # if hasattr(decoder.mid_block, 'attentions'):
        #     for attn in decoder.mid_block.attentions:
        #         shard_attention_block(attn, tp_degree)
    
    # up_blocks暂时不分片，避免复杂性
    # 或者可以选择性地只分片某些层
    
    # conv_out保持原样
    
    return decoder

def get_decoder_model(tp_degree: int):
    """获取并分片decoder模型"""
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32, 
        cache_dir="wan2.1_t2v_hf_cache_dir"
    )
    
    decoder = vae.decoder
    decoder.eval()
    
    # 先上转精度
    upcast_norms_to_f32(decoder)
    
    # 简化的分片策略
    decoder = shard_decoder(decoder, tp_degree)
    
    # post_quant_conv不分片
    
    # 创建包装器
    decoder_wrapper = DecoderWrapper(decoder, vae.post_quant_conv)
    
    return decoder_wrapper, {}

def compile_decoder(args):
    tp_degree = 8  # 使用8个tensor并行度
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    
    latent_height = args.height // 8
    latent_width = args.width // 8
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    batch_size = 1
    frames = args.frames  # 现在可以设置为21
    in_channels = 16
    
    sample_inputs = torch.rand(
        (batch_size, in_channels, frames, latent_height, latent_width), 
        dtype=torch.float32
    )
    
    get_decoder_model_f = partial(get_decoder_model, tp_degree)
    
    with torch.no_grad():
        compiled_decoder = neuronx_distributed.trace.parallel_model_trace(
            get_decoder_model_f,
            (sample_inputs,),
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )
        
        compiled_model_dir = f"{compiled_models_dir}/decoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        
        neuronx_distributed.trace.parallel_model_save(
            compiled_decoder, f"{compiled_model_dir}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video.", type=int, default=256)
    parser.add_argument("--width", help="width of generated video.", type=int, default=256)
    parser.add_argument("--frames", help="number of frames.", type=int, default=21)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_decoder(args)