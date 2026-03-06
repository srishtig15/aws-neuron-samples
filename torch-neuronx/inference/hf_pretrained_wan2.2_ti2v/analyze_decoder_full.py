"""
分析完整WanDecoder3d的结构（包含up_blocks），确定feat_cache需要的tensor形状
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

# 这次不删除up_blocks，保留完整decoder
print(f"decoder配置:")
print(f"  dim={decoder.dim}")
print(f"  z_dim={decoder.z_dim}")
print(f"  dim_mult={decoder.dim_mult}")
print(f"  num_res_blocks={decoder.num_res_blocks}")
print(f"  temperal_upsample={decoder.temperal_upsample}")

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
print("\n" + "="*80)
print("追踪feat_cache的使用情况:")
print("="*80)

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
        output1 = decoder(sample_input, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=True)
        print(f"输出形状: {list(output1.shape)}")
        print(f"feat_idx最终值: {feat_idx[0]}")
    except Exception as e:
        print(f"出错: {e}")
        import traceback
        traceback.print_exc()

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
        import traceback
        traceback.print_exc()

# 打印feat_cache中每个元素的形状
print("\n" + "="*80)
print("feat_cache中各tensor的形状:")
print("="*80)
print(f"{'索引':<5} {'形状':<30} {'层名称'}")
print("-" * 80)
for i, cache in enumerate(feat_cache):
    if cache is None:
        shape_str = "None"
    elif isinstance(cache, str):
        shape_str = f"'{cache}' (字符串)"
    else:
        shape_str = str(list(cache.shape))

    layer_name = conv_layers[i][0] if i < len(conv_layers) else "unknown"
    print(f"{i:<5} {shape_str:<30} {layer_name}")

# 按空间分辨率分组
print("\n" + "="*80)
print("按空间分辨率分组:")
print("="*80)

resolution_groups = {}
for i, cache in enumerate(feat_cache):
    if cache is not None and not isinstance(cache, str):
        h, w = cache.shape[3], cache.shape[4]
        key = f"{h}x{w}"
        if key not in resolution_groups:
            resolution_groups[key] = []
        resolution_groups[key].append((i, cache.shape, conv_layers[i][0]))

for resolution, items in sorted(resolution_groups.items()):
    print(f"\n分辨率 {resolution}:")
    for idx, shape, name in items:
        print(f"  {idx}: {list(shape)} - {name}")

# 生成用于trace的feat_cache代码
print("\n" + "="*80)
print("用于torch_neuronx.trace的feat_cache代码:")
print("="*80)

print("feat_cache = [")
for i, cache in enumerate(feat_cache):
    layer_name = conv_layers[i][0] if i < len(conv_layers) else "unknown"

    if cache is None:
        print(f"    None,  # {i}: {layer_name}")
    elif isinstance(cache, str):
        print(f"    None,  # {i}: {layer_name} (初始为'{cache}')")
    else:
        # 计算实际的空间尺寸因子
        spatial_factor_h = cache.shape[3] // latent_height
        spatial_factor_w = cache.shape[4] // latent_width

        if spatial_factor_h == 1:
            h_str = "latent_height"
        else:
            h_str = f"latent_height*{spatial_factor_h}"

        if spatial_factor_w == 1:
            w_str = "latent_width"
        else:
            w_str = f"latent_width*{spatial_factor_w}"

        shape_str = f"batch_size, {cache.shape[1]}, {cache.shape[2]}, {h_str}, {w_str}"
        print(f"    torch.rand(({shape_str}), dtype=torch.float32),  # {i}: {layer_name}")

print("]")

# 打印完整的trace代码建议
print("\n" + "="*80)
print("完整的trace代码建议:")
print("="*80)

print("""
# 在compile_decoder.py中替换第64-77行为:

        sample_inputs_1 = torch.rand((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.float32)
        feat_cache = [""")

for i, cache in enumerate(feat_cache):
    layer_name = conv_layers[i][0] if i < len(conv_layers) else "unknown"

    if cache is None:
        print(f"            None,  # {i}: {layer_name}")
    elif isinstance(cache, str):
        print(f"            None,  # {i}: {layer_name}")
    else:
        # 计算实际的空间尺寸因子
        spatial_factor_h = cache.shape[3] // latent_height
        spatial_factor_w = cache.shape[4] // latent_width

        if spatial_factor_h == 1:
            h_str = "latent_height"
        else:
            h_str = f"latent_height*{spatial_factor_h}"

        if spatial_factor_w == 1:
            w_str = "latent_width"
        else:
            w_str = f"latent_width*{spatial_factor_w}"

        shape_str = f"batch_size, {cache.shape[1]}, {cache.shape[2]}, {h_str}, {w_str}"
        print(f"            torch.rand(({shape_str}), dtype=torch.float32),  # {i}: {layer_name}")

print("""        ]
        feat_idx = [0]

        compiled_decoder = torch_neuronx.trace(
            decoder,
            (sample_inputs_1, feat_cache, feat_idx),  # 注意：这里传入tuple
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
""")

# 统计内存使用
print("\n" + "="*80)
print("内存占用估算:")
print("="*80)

total_elements = 0
for i, cache in enumerate(feat_cache):
    if cache is not None and not isinstance(cache, str):
        elements = cache.numel()
        total_elements += elements
        size_mb = elements * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"{i}: {elements:,} elements = {size_mb:.2f} MB - {conv_layers[i][0]}")

total_size_mb = total_elements * 4 / (1024 * 1024)
print(f"\n总计: {total_elements:,} elements = {total_size_mb:.2f} MB")
