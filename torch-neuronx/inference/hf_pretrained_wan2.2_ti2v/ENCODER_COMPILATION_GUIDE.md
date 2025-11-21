# VAE Encoder 编译指南

本文档说明如何编译和使用VAE Encoder用于图生视频（Image-to-Video）功能。

## 概述

对于图生视频（I2V），需要编译VAE encoder来处理输入图像。encoder将RGB图像编码为latent表示，供transformer处理。

## 文件说明

### 1. `compile_encoder.py` ✅ 已修正
**位置**: `neuron_wan2_2_ti2v/compile_encoder.py`

**主要修正内容:**
- ✅ 输入从latent改为RGB视频帧: `(batch_size, 3, frames, height, width)`
- ✅ feat_cache数量从34改为26（encoder有26个WanCausalConv3d层）
- ✅ feat_cache形状匹配encoder的结构（下采样: 512→256→128→64）
- ✅ 移除了错误的`upcast_norms_to_f32`（encoder是down_blocks不是up_blocks）
- ✅ 使用正确的channel数量: 12→160→320→640

**编译输入规格:**
```python
encoder_input: (1, 3, 2, 512, 512)  # RGB frames, CACHE_T=2
feat_cache: 26个元素的列表，每个元素shape为 (1, channels, 2, h, w)
```

### 2. `EncoderWrapper` ✅ 已实现
**位置**: `neuron_wan2_2_ti2v/neuron_commons.py`

**功能:**
- 处理TorchScript编译模型的feat_cache兼容性
- 将None值替换为零张量（TorchScript不支持None）
- 支持1帧输入（通过复制帧来满足CACHE_T=2要求）
- 正确传播feat_cache更新以支持时间缓存
- 无需调整输出（encoder自然输出正确的latent帧数）

## Encoder 结构分析

### 通道数变化
```
Input:  3 channels (RGB)
conv_in: 12 → 160
down_blocks.0: 160 (512x512)
down_blocks.1: 160 → 320 (256x256)
down_blocks.2: 320 → 640 (128x128)
down_blocks.3: 640 (64x64)
mid_block: 640 (64x64)
conv_out: 640 → 96 (z_dim*2 = 48*2)
```

### 时间下采样
```
Input frames: N
After down_blocks.1: N (no temporal downsample)
After down_blocks.2: N//2 (2x temporal downsample)
After down_blocks.3: N//4 (another 2x temporal downsample)
Output latent_frames: (N-1)//4 + 1
```

### Feat_cache 结构（26层）

```python
# 所有feat_cache的时间维度都是2 (CACHE_T=2)
feat_cache = [
    # 0: conv_in (12 channels, 512x512)
    (1, 12, 2, 512, 512),

    # 1-4: down_blocks.0 (160 channels, 512x512)
    (1, 160, 2, 512, 512),  # resnets.0.conv1
    (1, 160, 2, 512, 512),  # resnets.0.conv2
    (1, 160, 2, 512, 512),  # resnets.1.conv1
    (1, 160, 2, 512, 512),  # resnets.1.conv2

    # 5-10: down_blocks.1 (320 channels, 256x256)
    (1, 320, 2, 256, 256),  # resnets.0.conv1
    (1, 320, 2, 256, 256),  # resnets.0.conv2
    (1, 320, 2, 256, 256),  # resnets.0.conv_shortcut
    (1, 320, 2, 256, 256),  # resnets.1.conv1
    (1, 320, 2, 256, 256),  # resnets.1.conv2
    (1, 320, 2, 256, 256),  # downsampler.time_conv

    # 11-16: down_blocks.2 (640 channels, 128x128)
    (1, 640, 2, 128, 128),  # resnets.0.conv1
    (1, 640, 2, 128, 128),  # resnets.0.conv2
    (1, 640, 2, 128, 128),  # resnets.0.conv_shortcut
    (1, 640, 2, 128, 128),  # resnets.1.conv1
    (1, 640, 2, 128, 128),  # resnets.1.conv2
    (1, 640, 2, 128, 128),  # downsampler.time_conv

    # 17-20: down_blocks.3 (640 channels, 64x64)
    (1, 640, 2, 64, 64),    # resnets.0.conv1
    (1, 640, 2, 64, 64),    # resnets.0.conv2
    (1, 640, 2, 64, 64),    # resnets.1.conv1
    (1, 640, 2, 64, 64),    # resnets.1.conv2

    # 21-24: mid_block (640 channels, 64x64)
    (1, 640, 2, 64, 64),    # resnets.0.conv1
    (1, 640, 2, 64, 64),    # resnets.0.conv2
    (1, 640, 2, 64, 64),    # resnets.1.conv1
    (1, 640, 2, 64, 64),    # resnets.1.conv2

    # 25: conv_out (640 channels, 64x64)
    (1, 640, 2, 64, 64),    # conv_out
]
```

## 编译步骤

### 1. 编译Encoder

```bash
# 对于512x512分辨率
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512

# 对于256x256分辨率
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 256 \
    --width 256
```

**输出:**
- `compile_workdir_latency_optimized/encoder/model.pt`: 编译后的模型

### 2. 在推理中使用

```python
from neuron_wan2_2_ti2v.neuron_commons import EncoderWrapper
import torch
import torch.jit

# 加载pipeline
from diffusers import WanPipeline, AutoencoderKLWan

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir="wan2.2_ti2v_hf_cache_dir"
)
pipe = WanPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
    cache_dir="wan2.2_ti2v_hf_cache_dir"
)

# 使用编译后的encoder
encoder_model_path = "compile_workdir_latency_optimized/encoder/model.pt"

vae_encoder_wrapper = EncoderWrapper(pipe.vae.encoder)
# Encoder CANNOT use DataParallel because it accepts feat_cache (list argument)
# Use EncoderWrapper to handle TorchScript's feat_cache compatibility (None -> zero tensors)
vae_encoder_wrapper.model = torch.jit.load(encoder_model_path)
pipe.vae.encoder = vae_encoder_wrapper

# 现在可以使用I2V功能
from PIL import Image

init_image = Image.open("cat.jpg").resize((512, 512))

output = pipe(
    image=init_image,  # 输入图像
    prompt="A cat walks on the grass, realistic",
    height=512,
    width=512,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50
).frames[0]
```

## 关键差异：Encoder vs Decoder

| 特性 | Encoder | Decoder |
|------|---------|---------|
| **输入** | RGB视频帧 (3 channels) | Latent (48 channels) |
| **输出** | Latent (48 channels) | RGB视频帧 (3 channels) |
| **空间变化** | 下采样 8x (512→64) | 上采样 8x (64→512) |
| **时间变化** | 下采样 4x | 上采样 4x |
| **feat_cache数量** | 26层 | 34层 |
| **主结构** | down_blocks | up_blocks |
| **DataParallel** | 不支持（list参数） | 不支持（list参数） |
| **Wrapper** | EncoderWrapper | DecoderWrapper |

## 工作流程：I2V vs T2V

### Text-to-Video (T2V)
```
Text → Text Encoder → Transformer → VAE Decoder → Video
```

### Image-to-Video (I2V)
```
Image → VAE Encoder → Transformer → VAE Decoder → Video
           ↓
        Latent
```

### 完整I2V流程
```python
# 1. Encode image to latent
image_input = torch.randn(1, 3, 5, 512, 512)  # 5 frames
latents = vae.encode(image_input).latent_dist.sample()
# shape: (1, 48, 2, 64, 64)  # 5 frames → 2 latent frames

# 2. Transformer processes latent
transformer_output = transformer(latents, text_embeddings, ...)
# shape: (1, 48, 21, 64, 64)  # 扩展到21帧

# 3. Decode to video
video_output = vae.decode(transformer_output).sample
# shape: (1, 3, 84, 512, 512)  # 21帧 → 84帧
```

## 编译参数说明

```python
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"     # trn2
compiler_flags = """
    --verbose=INFO
    --target=trn2
    --lnc=2                                       # 使用2个逻辑NeuronCore
    --model-type=unet-inference                   # UNet类型推理优化
    --enable-fast-loading-neuron-binaries         # 快速加载
"""
```

## 内存估算

### Encoder feat_cache 内存占用

对于512x512分辨率：
```python
# 计算总内存
total_elements = 0
for shape in feat_cache_shapes:
    b, c, t, h, w = shape
    total_elements += b * c * t * h * w

# float32: 4 bytes per element
memory_mb = total_elements * 4 / (1024 * 1024)
# 约 1100 MB for encoder feat_cache
```

### 对比

| 组件 | feat_cache层数 | 内存占用 (512x512) |
|------|---------------|-------------------|
| Encoder | 26 | ~1100 MB |
| Decoder | 34 | ~1918 MB |
| **总计** | 60 | ~3018 MB |

## 常见问题

### Q1: 为什么不能用DataParallel？
**A:** Encoder/Decoder的forward方法接受`feat_cache`参数，类型是`list[Tensor]`。PyTorch的DataParallel无法正确scatter list参数。

### Q2: 为什么要用EncoderWrapper？
**A:** TorchScript编译的模型不支持None值。EncoderWrapper将feat_cache中的None替换为零张量，使其与TorchScript兼容。

### Q3: 为什么编译时用2帧？
**A:** 因为CACHE_T=2，所有WanCausalConv3d层需要缓存最近2帧。编译时使用2帧确保所有路径都被编译。

### Q4: 1帧输入会怎样？
**A:** EncoderWrapper会将1帧复制为2帧以满足CACHE_T=2要求。encoder会正常处理，输出正确的latent帧数。

### Q5: Encoder的时间下采样是怎么工作的？
**A:** Encoder有两个时间下采样层（stride=2的time_conv）:
- down_blocks.1.downsampler.time_conv: 第一次2x下采样
- down_blocks.2.downsampler.time_conv: 第二次2x下采样
- 总计: 4x时间下采样

### Q6: 如何验证编译是否正确？
**A:** 运行测试：
```python
# 测试feat_cache传播
encoder_wrapper = EncoderWrapper(encoder)
encoder_wrapper.model = torch.jit.load("compiled_model.pt")

# 创建测试输入
test_input = torch.randn(1, 3, 5, 512, 512, dtype=torch.float32)
vae.clear_cache()

# 运行编码
output = vae.encode(test_input).latent_dist.sample()

# 检查feat_cache是否被填充
non_none = sum(1 for c in vae._enc_feat_map if c is not None)
print(f"feat_cache filled: {non_none}/26")  # 应该是 26/26
```

## 下一步

完成encoder编译后，还需要：

1. **编译其他组件** (如果尚未完成):
   - Text Encoder
   - Transformer
   - Decoder
   - post_quant_conv

2. **集成到推理脚本**:
   - 在`run_wan2.2_ti2v_latency_optimized.py`中加载编译后的encoder
   - 支持I2V输入

3. **测试**:
   - 验证I2V生成质量
   - 对比CPU/GPU结果
   - 性能基准测试

## 参考资料

- 原始decoder编译: `compile_decoder.py`
- Encoder分析脚本: `/tmp/analyze_encoder_cache.py`
- VAE实现: `diffusers/models/autoencoders/autoencoder_kl_wan.py`
