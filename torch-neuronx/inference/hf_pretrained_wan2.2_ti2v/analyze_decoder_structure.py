"""
分析WanDecoder3d的结构，确定feat_cache需要的tensor形状
"""
import os
os.environ["NEURON_INTERNAL_USE_VANILLA_TORCH_XLA"] = "1"

from diffusers import AutoencoderKLWan
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
)

decoder = vae.decoder
decoder.eval()

# 删除up_blocks部分（跟compile_decoder.py保持一致）
del decoder.up_blocks
del decoder.norm_out
del decoder.conv_out

# 统计WanCausalConv3d层
conv_layers = []
for name, module in decoder.named_modules():
    if isinstance(module, WanCausalConv3d):
        conv_layers.append((name, module))

print(f"\n总共有 {len(conv_layers)} 个WanCausalConv3d层:\n")
for i, (name, module) in enumerate(conv_layers):
    print(f"{i}: {name}")
    print(f"   in_channels={module.in_channels}, out_channels={module.out_channels}")
    print(f"   kernel_size={module.kernel_size}")

# 模拟forward pass来追踪feat_cache的使用
print("\n" + "="*60)
print("追踪feat_cache的使用情况:")
print("="*60)

batch_size = 1
in_channels = 48  # decoder.conv_in的输入通道数
frames = 1  # 单帧测试
latent_height = 64  # 512 // 8
latent_width = 64

# 准备输入和feat_cache
sample_input = torch.rand((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.float32)

# 初始化feat_cache（全部为None）
feat_cache = [None] * len(conv_layers)
feat_idx = [0]

print(f"\n输入形状: {list(sample_input.shape)}")
print(f"feat_cache初始长度: {len(feat_cache)}\n")

# 第一次Forward pass（初始化feat_cache）
print("第一次forward pass（初始化feat_cache）:")
with torch.no_grad():
    try:
        feat_idx[0] = 0
        output1 = decoder(sample_input, feat_cache=feat_cache, feat_idx=feat_idx)
        print(f"输出形状: {list(output1.shape)}")
        print(f"feat_idx最终值: {feat_idx[0]}")
    except Exception as e:
        print(f"出错: {e}")

# 第二次Forward pass（使用feat_cache）
print("\n第二次forward pass（使用缓存）:")
sample_input2 = torch.rand((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.float32)
with torch.no_grad():
    try:
        feat_idx[0] = 0
        output2 = decoder(sample_input2, feat_cache=feat_cache, feat_idx=feat_idx)
        print(f"输出形状: {list(output2.shape)}")
        print(f"feat_idx最终值: {feat_idx[0]}")
    except Exception as e:
        print(f"出错: {e}")

# 打印feat_cache中每个元素的形状
print("\n" + "="*60)
print("feat_cache中各tensor的形状:")
print("="*60)
for i, cache in enumerate(feat_cache):
    if cache is None:
        print(f"{i}: None")
    elif isinstance(cache, str):
        print(f"{i}: '{cache}' (字符串)")
    else:
        print(f"{i}: {list(cache.shape)} - {conv_layers[i][0]}")

# 生成用于trace的feat_cache代码
print("\n" + "="*60)
print("用于torch_neuronx.trace的feat_cache代码:")
print("="*60)
print("feat_cache = [")
for i, cache in enumerate(feat_cache):
    if cache is None:
        print(f"    None,  # {i}: {conv_layers[i][0]}")
    elif isinstance(cache, str):
        print(f"    None,  # {i}: {conv_layers[i][0]} (初始为'{cache}')")
    else:
        shape_str = f"({batch_size}, {cache.shape[1]}, {cache.shape[2]}, latent_height, latent_width)"
        print(f"    torch.rand({shape_str}, dtype=torch.float32),  # {i}: {conv_layers[i][0]}")
print("]")

# 打印完整的trace代码建议
print("\n" + "="*60)
print("完整的trace建议（替换compile_decoder.py中的第64-77行）:")
print("="*60)

print("""
        sample_inputs_1 = torch.rand((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.float32)
        feat_cache = [""")

for i, cache in enumerate(feat_cache):
    if cache is None:
        print(f"            None,  # {i}: {conv_layers[i][0]}")
    elif isinstance(cache, str):
        print(f"            None,  # {i}: {conv_layers[i][0]}")
    else:
        # 计算实际的空间尺寸因子
        spatial_factor = cache.shape[3] // latent_height
        if spatial_factor == 1:
            h_str = "latent_height"
            w_str = "latent_width"
        else:
            h_str = f"latent_height*{spatial_factor}"
            w_str = f"latent_width*{spatial_factor}"

        shape_str = f"batch_size, {cache.shape[1]}, {cache.shape[2]}, {h_str}, {w_str}"
        print(f"            torch.rand(({shape_str}), dtype=torch.float32),  # {i}: {conv_layers[i][0]}")

print("""        ]
        feat_idx = [0]

        compiled_decoder = torch_neuronx.trace(
            decoder,
            (sample_inputs_1, feat_cache, feat_idx),
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
""")
