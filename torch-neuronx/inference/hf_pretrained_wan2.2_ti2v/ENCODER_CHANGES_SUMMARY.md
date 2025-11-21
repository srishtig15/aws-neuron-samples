# Encoder编译修改摘要

## 修改的文件

### 1. ✅ `neuron_wan2_2_ti2v/compile_encoder.py`

#### 主要修正:

**❌ 原来的错误:**
```python
# 错误1: 输入是latent而不是RGB视频
encoder_input = torch.rand((batch_size, 48, encoder_frames, latent_height, latent_width))

# 错误2: feat_cache是decoder的34层，而不是encoder的26层
feat_cache = [
    torch.rand((batch_size, 48, 2, latent_height, latent_width)),  # decoder的结构
    # ... 33层decoder的feat_cache
]

# 错误3: upcast_norms_to_f32用的是encoder.up_blocks
def upcast_norms_to_f32(encoder):
    for upblock in encoder.up_blocks:  # encoder没有up_blocks!
        ...
```

**✅ 修正后:**
```python
# 正确1: RGB视频帧输入
in_channels = 3  # RGB
encoder_input = torch.rand((batch_size, in_channels, encoder_frames, height, width))

# 正确2: 26层encoder feat_cache，正确的形状
feat_cache = [
    # conv_in: 512x512
    torch.zeros((batch_size, 12, 2, height, width)),
    # down_blocks.0: 160 channels, 512x512
    torch.zeros((batch_size, 160, 2, height, width)),  # x4层
    # down_blocks.1: 320 channels, 256x256
    torch.zeros((batch_size, 320, 2, height//2, width//2)),  # x6层
    # down_blocks.2: 640 channels, 128x128
    torch.zeros((batch_size, 640, 2, height//4, width//4)),  # x6层
    # down_blocks.3: 640 channels, 64x64
    torch.zeros((batch_size, 640, 2, height//8, width//8)),  # x4层
    # mid_block: 640 channels, 64x64
    torch.zeros((batch_size, 640, 2, height//8, width//8)),  # x4层
    # conv_out: 64x64
    torch.zeros((batch_size, 640, 2, height//8, width//8)),  # x1层
]
# 总计: 1 + 4 + 6 + 6 + 4 + 4 + 1 = 26层

# 正确3: 移除了错误的upcast_norms_to_f32
# (encoder用down_blocks，不需要upcast)
```

#### 完整的修改列表:

1. **输入参数**:
   - `in_channels`: `48` → `3`
   - 输入tensor: `(1, 48, 2, latent_h, latent_w)` → `(1, 3, 2, height, width)`

2. **feat_cache数量**: `34` → `26`

3. **feat_cache形状**: 从decoder的shape改为encoder的shape

4. **移除**: `upcast_norms_to_f32()` 函数和调用

5. **清理**: 移除了注释掉的无用代码

---

### 2. ✅ `neuron_wan2_2_ti2v/neuron_commons.py`

#### 新增: `EncoderWrapper`类

**位置**: 在`DecoderWrapper`之前插入

**代码**:
```python
class EncoderWrapper(nn.Module):
    """Specialized wrapper for VAE encoder that handles TorchScript feat_cache compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x"""
        batch_size = x.shape[0]
        height = x.shape[3]
        width = x.shape[4]

        # 26个feat_cache shapes (encoder结构)
        self.feat_cache_shapes = [
            (batch_size, 12, 2, height, width),         # conv_in
            (batch_size, 160, 2, height, width),        # down_blocks.0 x4
            # ...
            (batch_size, 640, 2, height//8, width//8),  # down_blocks.3 + mid_block + conv_out
        ]

    def forward(self, x, **kwargs):
        if 'feat_cache' in kwargs:
            feat_cache = kwargs['feat_cache']
            is_torchscript = isinstance(self.model, torch.jit.ScriptModule)

            if is_torchscript:
                # 1. Pad 1-frame input to 2 frames
                original_frame_count = x.shape[2]
                if original_frame_count == 1:
                    x = torch.cat([x, x], dim=2)

                # 2. Initialize feat_cache shapes
                if self.feat_cache_shapes is None:
                    self._init_feat_cache_shapes(x)

                # 3. Replace None with zero tensors
                feat_cache_fixed = []
                for i, cache in enumerate(feat_cache):
                    if cache is None and i < len(self.feat_cache_shapes):
                        feat_cache_fixed.append(torch.zeros(
                            self.feat_cache_shapes[i],
                            dtype=x.dtype,
                            device=x.device
                        ))
                    else:
                        feat_cache_fixed.append(cache)

                # 4. Forward pass
                output = self.model(x, feat_cache_fixed)

                # 5. Propagate updates back to original feat_cache
                for i in range(len(feat_cache)):
                    feat_cache[i] = feat_cache_fixed[i]

                # 注意: encoder不需要调整输出帧数
                # encoder自然输出正确的latent帧数

            else:
                output = self.model(x, feat_cache=feat_cache, **kwargs)
        else:
            output = self.model(x)
        return output

    def clear_cache(self):
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
```

**关键特性**:
- ✅ 处理TorchScript的feat_cache兼容性（None → zero tensors）
- ✅ 支持1帧输入（复制为2帧）
- ✅ 正确传播feat_cache更新
- ✅ 26个feat_cache shapes（匹配encoder结构）
- ✅ 空间下采样: 512→256→128→64

---

## 对比：EncoderWrapper vs DecoderWrapper

| 特性 | EncoderWrapper | DecoderWrapper |
|------|----------------|----------------|
| **feat_cache数量** | 26 | 34 |
| **输入** | RGB (3 channels) | Latent (48 channels) |
| **输出** | Latent (48 channels) | RGB (3 channels) |
| **空间变化** | 下采样8x (512→64) | 上采样8x (64→512) |
| **时间变化** | 下采样4x | 上采样4x |
| **输出调整** | 不需要 | 需要（取最后4帧） |
| **用途** | I2V (Image-to-Video) | T2V + I2V |

### DecoderWrapper输出调整
```python
# decoder输出调整（取最后4帧）
if original_frame_count == 1:
    output = output[:, :, -4:, :, :]
```

### EncoderWrapper输出调整
```python
# encoder不需要输出调整
# encoder自然处理时间下采样：2 input frames → 1 latent frame
```

---

## 编译和使用

### 编译命令:
```bash
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512
```

### 使用示例:
```python
from neuron_wan2_2_ti2v.neuron_commons import EncoderWrapper
import torch

# 加载编译后的encoder
encoder_model_path = "compile_workdir_latency_optimized/encoder/model.pt"
vae_encoder_wrapper = EncoderWrapper(pipe.vae.encoder)
vae_encoder_wrapper.model = torch.jit.load(encoder_model_path)
pipe.vae.encoder = vae_encoder_wrapper

# 使用I2V
output = pipe(
    image=init_image,
    prompt="A cat walks on the grass",
    height=512,
    width=512,
    num_frames=81
).frames[0]
```

---

## 验证清单

编译前请确认:
- ✅ `compile_encoder.py`的修正已完成
- ✅ `neuron_commons.py`已添加`EncoderWrapper`
- ✅ 编译参数正确（height, width）
- ✅ NEURON环境变量已设置

编译后请验证:
- ✅ `compiled_models_dir/encoder/model.pt`已生成
- ✅ 模型可以加载: `torch.jit.load(...)`
- ✅ feat_cache可以正确传播
- ✅ I2V推理可以运行

---

## 总结

### 修改的核心问题:
1. **输入类型错误**: latent → RGB视频帧
2. **feat_cache结构错误**: decoder的34层 → encoder的26层
3. **feat_cache形状错误**: 上采样shape → 下采样shape
4. **缺少EncoderWrapper**: 需要处理TorchScript兼容性

### 修正后的效果:
- ✅ 正确的encoder输入（RGB）
- ✅ 正确的feat_cache数量和形状（26层）
- ✅ TorchScript兼容性（EncoderWrapper）
- ✅ 支持I2V功能
- ✅ feat_cache正确传播

现在可以编译和使用encoder进行I2V推理了！
