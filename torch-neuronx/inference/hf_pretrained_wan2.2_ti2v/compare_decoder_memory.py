"""
对比不同 num_frames 设置下的 decoder 内存占用
"""

def calculate_memory(height, width, num_frames):
    """计算给定参数下的内存占用"""
    latent_height = height // 16
    latent_width = width // 16
    latent_frames = (num_frames - 1) // 4 + 1

    batch_size = 1
    in_channels = 48
    CACHE_T = 2

    # Decoder 输入
    decoder_input_mb = batch_size * in_channels * latent_frames * latent_height * latent_width * 4 / (1024**2)

    # feat_cache layers
    feat_cache_layers = [
        (48, 1), (1024, 1), (1024, 1), (1024, 1), (1024, 1),
        (1024, 1), (1024, 1), (1024, 1), (1024, 1), (1024, 1),
        (1024, 1), (1024, 1), (1024, 2), (1024, 2), (1024, 2),
        (1024, 2), (1024, 2), (1024, 2), (1024, 2), (1024, 4),
        (512, 4), (512, 4), (512, 4), (512, 4), (512, 4),
        (512, 8), (256, 8), (256, 8), (256, 8), (256, 8),
        (256, 8), (256, 8), (256, 8), (12, 8)
    ]

    total_cache_mb = 0
    for channels, spatial_factor in feat_cache_layers:
        h = latent_height * spatial_factor
        w = latent_width * spatial_factor
        memory_mb = batch_size * channels * CACHE_T * h * w * 4 / (1024**2)
        total_cache_mb += memory_mb

    # Decoder 输出
    output_channels = 12
    decoder_output_mb = batch_size * output_channels * latent_frames * (latent_height * 8) * (latent_width * 8) * 4 / (1024**2)

    total_mb = decoder_input_mb + total_cache_mb + decoder_output_mb

    return {
        'latent_frames': latent_frames,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'decoder_input_mb': decoder_input_mb,
        'feat_cache_mb': total_cache_mb,
        'decoder_output_mb': decoder_output_mb,
        'total_mb': total_mb
    }

# 对比不同配置
configs = [
    (512, 512, 4),
    (512, 512, 7),
    (512, 512, 15),
    (512, 512, 31),
]

print("="*100)
print("不同 num_frames 设置下的 Decoder 内存占用对比")
print("="*100)
print()

for height, width, num_frames in configs:
    result = calculate_memory(height, width, num_frames)

    print(f"配置: height={height}, width={width}, num_frames={num_frames}")
    print(f"  latent_frames: {result['latent_frames']}")
    print(f"  Decoder 输入: {result['decoder_input_mb']:.2f} MB")
    print(f"  Feat_cache:   {result['feat_cache_mb']:.2f} MB")
    print(f"  Decoder 输出: {result['decoder_output_mb']:.2f} MB")
    print(f"  总计:        {result['total_mb']:.2f} MB ({result['total_mb']/1024:.2f} GB)")
    print()

print("="*100)
print("关键观察:")
print("="*100)
print()

# 分析 num_frames 变化的影响
result_7 = calculate_memory(512, 512, 7)
result_15 = calculate_memory(512, 512, 15)

input_ratio = result_15['decoder_input_mb'] / result_7['decoder_input_mb']
output_ratio = result_15['decoder_output_mb'] / result_7['decoder_output_mb']
cache_ratio = result_15['feat_cache_mb'] / result_7['feat_cache_mb']

print(f"从 num_frames=7 增加到 num_frames=15:")
print(f"  - latent_frames: {result_7['latent_frames']} -> {result_15['latent_frames']} (增加 {result_15['latent_frames'] - result_7['latent_frames']} 帧)")
print(f"  - Decoder 输入内存: {result_7['decoder_input_mb']:.2f} MB -> {result_15['decoder_input_mb']:.2f} MB (增加 {input_ratio:.2f}x)")
print(f"  - Decoder 输出内存: {result_7['decoder_output_mb']:.2f} MB -> {result_15['decoder_output_mb']:.2f} MB (增加 {output_ratio:.2f}x)")
print(f"  - Feat_cache 内存: {result_7['feat_cache_mb']:.2f} MB -> {result_15['feat_cache_mb']:.2f} MB (增加 {cache_ratio:.2f}x)")
print()
print(f"⚠️  关键发现:")
print(f"  - feat_cache 是固定大小 ({result_15['feat_cache_mb']:.2f} MB)，不随 num_frames 变化")
print(f"  - 只有 decoder 输入和输出会随 latent_frames 线性增长")
print(f"  - feat_cache 占总内存的 {result_15['feat_cache_mb']/result_15['total_mb']*100:.1f}%")
print()

print("="*100)
print("瓶颈分析:")
print("="*100)
print()
print(f"1. feat_cache 是最大的内存占用，约 {result_15['feat_cache_mb']:.2f} MB (~{result_15['feat_cache_mb']/1024:.2f} GB)")
print(f"2. feat_cache 是固定大小，因为 CACHE_T=2 (缓存帧数固定)")
print(f"3. 后期的 up_blocks (特别是 up_blocks.2 和 up_blocks.3) 占用最多内存")
print(f"   - 因为空间分辨率增大 (从 32x32 到 256x256)")
print(f"4. 增加 num_frames 主要影响:")
print(f"   - Decoder 的输入/输出 tensor 大小")
print(f"   - 不影响 feat_cache 大小")
print()

print("="*100)
print("优化建议:")
print("="*100)
print()
print("1. 减少 feat_cache 内存占用的方法:")
print("   - 降低空间分辨率 (height/width)")
print("   - 使用更小的通道数 (需要修改模型架构)")
print("   - 使用低精度 (bfloat16/float16 代替 float32)")
print()
print("2. 当前 decoder 使用 float32，如果改用 bfloat16:")
print(f"   - feat_cache 将减少到 {result_15['feat_cache_mb']/2:.2f} MB")
print(f"   - 总内存将减少到 {result_15['total_mb']/2:.2f} MB (~{result_15['total_mb']/2/1024:.2f} GB)")
print()
print("3. num_frames=15 相比 num_frames=7:")
print(f"   - 总内存仅增加 {result_15['total_mb'] - result_7['total_mb']:.2f} MB")
print(f"   - 相对增幅很小 (因为 feat_cache 不变)")
