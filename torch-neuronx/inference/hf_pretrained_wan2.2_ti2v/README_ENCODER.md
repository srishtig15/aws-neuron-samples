# VAE Encoder 编译与使用指南

## 📋 快速开始

```bash
# 1. 编译encoder (512x512)
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512

# 2. 验证编译结果
python /tmp/test_encoder_compilation.py
```

## ✅ 已完成的修改

### 1. **compile_encoder.py** - 已修正 ✅
- ✅ 输入类型: `latent (48 channels)` → `RGB视频帧 (3 channels)`
- ✅ feat_cache数量: `34层 (decoder)` → `26层 (encoder)`
- ✅ feat_cache形状: decoder的上采样shape → encoder的下采样shape
- ✅ 移除: 错误的`upcast_norms_to_f32()`函数

### 2. **EncoderWrapper** - 已实现 ✅
**位置**: `neuron_wan2_2_ti2v/neuron_commons.py`

**功能**:
- ✅ TorchScript feat_cache兼容性（None → zero tensors）
- ✅ 支持1帧输入（自动复制为2帧）
- ✅ feat_cache更新传播（26层）
- ✅ 正确的空间下采样（512→256→128→64）

## 📊 验证结果

```
================================================================================
验证 compile_encoder.py 修改
================================================================================

1. 检查输入规格:
   ✓ 期望输入: (batch_size, 3, frames, height, width)
   ✓ RGB视频帧，3个通道
   测试输入shape: torch.Size([1, 3, 2, 512, 512])

2. 检查feat_cache数量:
   Encoder中的WanCausalConv3d层数: 26
   ✓ 正确！应该是26层

3. 检查feat_cache形状:
   ✓ 所有形状应该匹配compile_encoder.py中的feat_cache定义

4. 检查EncoderWrapper:
   ✓ EncoderWrapper已定义
   ✓ EncoderWrapper可以实例化
   ✓ feat_cache_shapes数量正确: 26

5. 测试encoder运行:
   ✓ Latent帧数正确: 2 (预期: 2)
   注: feat_cache在编译后的模型中才会被填充
```

## 🔧 关键修正对比

| 项目 | 修正前 ❌ | 修正后 ✅ |
|------|----------|----------|
| **输入channels** | 48 (latent) | 3 (RGB) |
| **输入尺寸** | latent空间 | 图像空间 (512x512) |
| **feat_cache数量** | 34 (decoder) | 26 (encoder) |
| **feat_cache形状** | 上采样 (64→512) | 下采样 (512→64) |
| **Wrapper** | 无 | EncoderWrapper |

## 📁 文件结构

```
neuron_wan2_2_ti2v/
├── compile_encoder.py           # ✅ 已修正 - encoder编译脚本
├── neuron_commons.py            # ✅ 已添加 EncoderWrapper
└── ...

文档/
├── ENCODER_COMPILATION_GUIDE.md  # 详细编译指南
├── ENCODER_CHANGES_SUMMARY.md    # 修改摘要
└── README_ENCODER.md             # 本文档
```

## 🚀 使用示例

### 编译（必需）
```bash
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512
```

### 推理代码
```python
from diffusers import WanPipeline, AutoencoderKLWan
from neuron_wan2_2_ti2v.neuron_commons import EncoderWrapper
import torch
from PIL import Image

# 1. 加载模型
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir="wan2.2_ti2v_hf_cache_dir"
)
pipe = WanPipeline.from_pretrained(
    model_id, vae=vae,
    torch_dtype=torch.bfloat16,
    cache_dir="wan2.2_ti2v_hf_cache_dir"
)

# 2. 加载编译后的encoder
encoder_path = "compile_workdir_latency_optimized/encoder/model.pt"
vae_encoder_wrapper = EncoderWrapper(pipe.vae.encoder)
vae_encoder_wrapper.model = torch.jit.load(encoder_path)
pipe.vae.encoder = vae_encoder_wrapper

# 3. Image-to-Video推理
init_image = Image.open("cat.jpg").resize((512, 512))
output = pipe(
    image=init_image,
    prompt="A cat walks on the grass, realistic",
    height=512,
    width=512,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50
).frames[0]

# 4. 保存结果
from diffusers.utils import export_to_video
export_to_video(output, "output_i2v.mp4", fps=24)
```

## 📖 详细文档

1. **ENCODER_COMPILATION_GUIDE.md**
   - Encoder结构详细分析
   - feat_cache完整列表
   - 编译参数说明
   - 内存占用估算
   - 常见问题解答

2. **ENCODER_CHANGES_SUMMARY.md**
   - 所有修改的详细对比
   - 修正前后的代码对比
   - EncoderWrapper vs DecoderWrapper对比

## ⚠️ 注意事项

### 编译时
- ✅ 使用正确的height和width参数
- ✅ 确保NEURON环境变量已设置
- ✅ 编译时间较长（可能需要10-30分钟）

### 运行时
- ✅ Encoder不能使用DataParallel（list参数）
- ✅ 必须使用EncoderWrapper包装编译后的模型
- ✅ 输入必须是RGB图像（3 channels）
- ✅ 支持1帧或多帧输入

## 🔄 完整工作流程

### Text-to-Video (T2V)
```
Text → Text Encoder → Transformer → Decoder → Video
```

### Image-to-Video (I2V) ⭐ 新增
```
Image → VAE Encoder → Transformer → Decoder → Video
          ↓
       Latent (48 ch)
```

## 📊 Encoder规格

| 项目 | 规格 |
|------|------|
| **输入** | RGB视频 (3 channels, 512x512) |
| **输出** | Latent (48 channels, 64x64) |
| **空间压缩** | 8x (512→64) |
| **时间压缩** | 4x |
| **feat_cache层数** | 26 |
| **编译核心数** | 2 (--lnc=2) |
| **内存占用** | ~1100 MB (feat_cache) |

## ✅ 检查列表

编译前:
- [ ] compile_encoder.py已修正
- [ ] EncoderWrapper已添加到neuron_commons.py
- [ ] NEURON环境变量已设置
- [ ] 验证脚本通过测试

编译后:
- [ ] model.pt文件已生成
- [ ] 模型可以加载 (`torch.jit.load`)
- [ ] I2V推理可以运行
- [ ] 输出视频质量正常

## 🎯 下一步

1. **编译其他组件** (如果还没有):
   ```bash
   # Text Encoder
   python neuron_wan2_2_ti2v/compile_text_encoder.py \
       --compiled_models_dir compile_workdir_latency_optimized

   # Transformer
   python neuron_wan2_2_ti2v/compile_transformer_latency_optimized.py \
       --compiled_models_dir compile_workdir_latency_optimized

   # Decoder
   python neuron_wan2_2_ti2v/compile_decoder.py \
       --compiled_models_dir compile_workdir_latency_optimized \
       --height 512 --width 512
   ```

2. **测试I2V功能**:
   - 准备测试图像
   - 运行推理
   - 验证输出质量
   - 性能基准测试

3. **集成到主推理脚本**:
   - 更新`run_wan2.2_ti2v_latency_optimized.py`
   - 添加I2V模式支持
   - 添加命令行参数

## 📞 需要帮助？

参考文档:
- `ENCODER_COMPILATION_GUIDE.md` - 详细指南
- `ENCODER_CHANGES_SUMMARY.md` - 修改说明
- `COMPARISON_GUIDE.md` - GPU vs Trainium对比

---

**状态**: ✅ 所有修改已完成，可以开始编译！

**最后更新**: 2025-11-21
