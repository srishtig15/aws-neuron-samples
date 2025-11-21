"""
计算 decoder 在 height=512, width=512, num_frames=15 情况下各层的形状
"""

# 输入参数
height = 512
width = 512
num_frames = 15

# 计算 latent 维度
# VAE 将空间维度降低 8 倍，transformer 的 patch_embedding 再降低 2 倍
latent_height = height // 16  # 512 // 16 = 32
latent_width = width // 16    # 512 // 16 = 32

# 计算 latent_frames
# 从 transformer 输出得到，公式: (num_frames - 1) // 4 + 1
latent_frames = (num_frames - 1) // 4 + 1  # (15 - 1) // 4 + 1 = 4

print("="*80)
print(f"输入参数: height={height}, width={width}, num_frames={num_frames}")
print("="*80)
print(f"Latent 空间维度:")
print(f"  latent_height = {latent_height}")
print(f"  latent_width = {latent_width}")
print(f"  latent_frames = {latent_frames}")
print()

batch_size = 1
in_channels = 48

# Decoder 输入形状
decoder_input_shape = (batch_size, in_channels, latent_frames, latent_height, latent_width)
print(f"Decoder 输入形状: {decoder_input_shape}")
print(f"  Tensor size: {batch_size * in_channels * latent_frames * latent_height * latent_width * 4 / (1024**2):.2f} MB (float32)")
print()

print("="*80)
print("Decoder 各层的 feat_cache 形状 (CACHE_T=2 用于时间维度缓存):")
print("="*80)

CACHE_T = 2  # 固定的时间缓存帧数

# 定义所有层的 feat_cache 形状
# 格式: (层名称, 通道数, 空间维度倍数)
feat_cache_layers = [
    ("conv_in", 48, 1),
    ("mid_block.resnets.0.conv1", 1024, 1),
    ("mid_block.resnets.0.conv2", 1024, 1),
    ("mid_block.resnets.1.conv1", 1024, 1),
    ("mid_block.resnets.1.conv2", 1024, 1),
    ("up_blocks.0.resnets.0.conv1", 1024, 1),
    ("up_blocks.0.resnets.0.conv2", 1024, 1),
    ("up_blocks.0.resnets.1.conv1", 1024, 1),
    ("up_blocks.0.resnets.1.conv2", 1024, 1),
    ("up_blocks.0.resnets.2.conv1", 1024, 1),
    ("up_blocks.0.resnets.2.conv2", 1024, 1),
    ("up_blocks.0.upsampler.time_conv", 1024, 1),
    ("up_blocks.1.resnets.0.conv1", 1024, 2),
    ("up_blocks.1.resnets.0.conv2", 1024, 2),
    ("up_blocks.1.resnets.1.conv1", 1024, 2),
    ("up_blocks.1.resnets.1.conv2", 1024, 2),
    ("up_blocks.1.resnets.2.conv1", 1024, 2),
    ("up_blocks.1.resnets.2.conv2", 1024, 2),
    ("up_blocks.1.upsampler.time_conv", 1024, 2),
    ("up_blocks.2.resnets.0.conv1", 1024, 4),
    ("up_blocks.2.resnets.0.conv2", 512, 4),
    ("up_blocks.2.resnets.0.conv_shortcut", 512, 4),
    ("up_blocks.2.resnets.1.conv1", 512, 4),
    ("up_blocks.2.resnets.1.conv2", 512, 4),
    ("up_blocks.2.resnets.2.conv1", 512, 4),
    ("up_blocks.2.resnets.2.conv2", 512, 8),
    ("up_blocks.3.resnets.0.conv1", 256, 8),
    ("up_blocks.3.resnets.0.conv2", 256, 8),
    ("up_blocks.3.resnets.0.conv_shortcut", 256, 8),
    ("up_blocks.3.resnets.1.conv1", 256, 8),
    ("up_blocks.3.resnets.1.conv2", 256, 8),
    ("up_blocks.3.resnets.2.conv1", 256, 8),
    ("up_blocks.3.resnets.2.conv2", 256, 8),
    ("conv_out", 12, 8),
]

total_cache_memory = 0

for idx, (layer_name, channels, spatial_factor) in enumerate(feat_cache_layers):
    h = latent_height * spatial_factor
    w = latent_width * spatial_factor
    shape = (batch_size, channels, CACHE_T, h, w)

    # 计算内存大小 (float32 = 4 bytes)
    memory_mb = batch_size * channels * CACHE_T * h * w * 4 / (1024**2)
    total_cache_memory += memory_mb

    print(f"{idx:2d}: {layer_name:40s} - shape: {shape}")
    print(f"     Memory: {memory_mb:8.2f} MB")

print("="*80)
print(f"Total feat_cache memory: {total_cache_memory:.2f} MB")
print()

# 输出形状
output_channels = 12  # VAE decoder 输出 RGB 帧 (3 channels) * 4 = 12? 或者其他配置
output_frames = latent_frames  # decoder 按帧处理
output_height = latent_height * 8  # decoder 将空间维度放大 8 倍
output_width = latent_width * 8
decoder_output_shape = (batch_size, output_channels, output_frames, output_height, output_width)

print(f"Decoder 输出形状: {decoder_output_shape}")
print(f"  Tensor size: {batch_size * output_channels * output_frames * output_height * output_width * 4 / (1024**2):.2f} MB (float32)")
print()

# 总结
print("="*80)
print("内存瓶颈分析:")
print("="*80)
print(f"1. Decoder 输入: {batch_size * in_channels * latent_frames * latent_height * latent_width * 4 / (1024**2):.2f} MB")
print(f"2. Feat_cache (34 layers): {total_cache_memory:.2f} MB")
print(f"3. Decoder 输出: {batch_size * output_channels * output_frames * output_height * output_width * 4 / (1024**2):.2f} MB")
print(f"4. 总计: {batch_size * in_channels * latent_frames * latent_height * latent_width * 4 / (1024**2) + total_cache_memory + batch_size * output_channels * output_frames * output_height * output_width * 4 / (1024**2):.2f} MB")
print()

# 按内存大小排序找出最大的几个层
print("="*80)
print("内存占用最大的 10 个 feat_cache 层:")
print("="*80)

cache_memory_list = []
for idx, (layer_name, channels, spatial_factor) in enumerate(feat_cache_layers):
    h = latent_height * spatial_factor
    w = latent_width * spatial_factor
    memory_mb = batch_size * channels * CACHE_T * h * w * 4 / (1024**2)
    cache_memory_list.append((idx, layer_name, channels, spatial_factor, h, w, memory_mb))

# 按内存大小降序排序
cache_memory_list.sort(key=lambda x: x[6], reverse=True)

for idx, layer_name, channels, spatial_factor, h, w, memory_mb in cache_memory_list[:10]:
    print(f"{idx:2d}: {layer_name:40s}")
    print(f"     Shape: ({batch_size}, {channels}, {CACHE_T}, {h}, {w}) - {memory_mb:.2f} MB")
