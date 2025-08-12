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
        if isinstance(norm.gamma, nn.Parameter):
            orig_channels = norm.gamma.shape[0]
            # 计算分片后的通道数
            sharded_channels = orig_channels // tp_degree
            # 分片gamma
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            start_idx = tp_rank * sharded_channels
            end_idx = start_idx + sharded_channels
            
            norm.gamma = nn.Parameter(norm.gamma[start_idx:end_idx].clone())
            
        # 处理bias - 检查它是否是tensor还是scalar
        if hasattr(norm, 'bias') and norm.bias is not None:
            if isinstance(norm.bias, (nn.Parameter, torch.Tensor)):
                # 如果bias是tensor，进行分片
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                sharded_channels = norm.bias.shape[0] // tp_degree
                start_idx = tp_rank * sharded_channels
                end_idx = start_idx + sharded_channels
                norm.bias = nn.Parameter(norm.bias[start_idx:end_idx].clone())
            # 如果bias是scalar（float），保持不变
            # else: pass
    
    return norm

def shard_resnet_block(resnet, tp_degree):
    """分片ResNet块 - 更简单的策略"""
    # 对于VAE decoder，我们采用更保守的分片策略
    # 只分片主要的卷积层，不触碰归一化层
    
    # 分片conv1 - 输出通道维度
    if hasattr(resnet, 'conv1') and isinstance(resnet.conv1, nn.Conv3d):
        out_channels = resnet.conv1.out_channels
        if out_channels % tp_degree == 0:
            resnet.conv1 = shard_conv3d(resnet.conv1, tp_degree, dim=0)
    
    # conv2暂时不分片，避免复杂的维度匹配问题
    # 或者可以采用all-reduce策略
    
    return resnet

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

class PartialShardingDecoder(nn.Module):
    """部分分片的Decoder包装器"""
    def __init__(self, decoder, tp_degree):
        super().__init__()
        self.decoder = decoder
        self.tp_degree = tp_degree
        
        # 只对特定的大层进行分片
        self._shard_large_layers()
    
    def _shard_large_layers(self):
        """只分片计算量大的层"""
        # 可以选择性地分片某些层
        # 例如，只分片mid_block的某些卷积
        if hasattr(self.decoder, 'mid_block') and self.decoder.mid_block is not None:
            if hasattr(self.decoder.mid_block, 'resnets'):
                for resnet in self.decoder.mid_block.resnets:
                    # 只分片conv1，保持其他层不变
                    if hasattr(resnet, 'conv1') and isinstance(resnet.conv1, nn.Conv3d):
                        if resnet.conv1.out_channels >= 384 and resnet.conv1.out_channels % self.tp_degree == 0:
                            # 创建一个简单的分片包装
                            resnet.conv1 = self._create_sharded_conv(resnet.conv1)
    
    def _create_sharded_conv(self, conv):
        """创建分片的卷积层"""
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        out_channels_per_rank = conv.out_channels // self.tp_degree
        
        new_conv = nn.Conv3d(
            in_channels=conv.in_channels,
            out_channels=out_channels_per_rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode
        )
        
        # 分片权重
        start_idx = tp_rank * out_channels_per_rank
        end_idx = start_idx + out_channels_per_rank
        new_conv.weight.data = conv.weight.data[start_idx:end_idx].clone()
        
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[start_idx:end_idx].clone()
        
        return new_conv
    
    def forward(self, x):
        return self.decoder(x)

def get_decoder_model(tp_degree: int):
    """获取decoder模型 - 简化版本"""
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32, 
        cache_dir="wan2.1_t2v_hf_cache_dir"
    )
    
    decoder = vae.decoder
    decoder.eval()
    
    # 上转精度
    upcast_norms_to_f32(decoder)
    
    # 使用部分分片策略
    # decoder = PartialShardingDecoder(decoder, tp_degree)
    
    # 为了简化，暂时不分片decoder，只依赖数据并行
    # 或者使用pipeline并行
    
    # 创建包装器
    decoder_wrapper = DecoderWrapper(decoder, vae.post_quant_conv)
    
    return decoder_wrapper, {}

def compile_decoder_pipeline(args):
    """使用pipeline并行而不是tensor并行"""
    latent_height = args.height // 8
    latent_width = args.width // 8
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    batch_size = 1
    frames = 21  # 每批处理21帧
    in_channels = 16
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32, 
        cache_dir="wan2.1_t2v_hf_cache_dir"
    )
    
    decoder = vae.decoder
    decoder.eval()
    upcast_norms_to_f32(decoder)
    
    with torch.no_grad():
        # 编译较小的批次
        sample_inputs = torch.rand(
            (batch_size, in_channels, frames, latent_height, latent_width), 
            dtype=torch.float32
        )
        
        import torch_neuronx
        
        # 编译decoder
        compiled_decoder = torch_neuronx.trace(
            decoder,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False
        )
        
        compiled_model_dir = f"{compiled_models_dir}/decoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        torch.jit.save(compiled_decoder, f"{compiled_model_dir}/model.pt")

        # 编译post_quant_conv
        compiled_post_quant_conv = torch_neuronx.trace(
            vae.post_quant_conv,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/post_quant_conv",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False
        )
        
        compiled_model_dir = f"{compiled_models_dir}/post_quant_conv"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)     
        torch.jit.save(compiled_post_quant_conv, f"{compiled_model_dir}/model.pt")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video.", type=int, default=256)
    parser.add_argument("--width", help="width of generated video.", type=int, default=256)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    
    # 使用pipeline策略而不是tensor并行
    compile_decoder_pipeline(args)