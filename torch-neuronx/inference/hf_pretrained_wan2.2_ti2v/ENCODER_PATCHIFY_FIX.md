# Encoder编译问题修复：Patchify

## 问题

编译时报错：
```
RuntimeError: ... f32[1,12,512,512] vs. f32[1,3,512,512]
```

## 根本原因

**Encoder期望patchify后的输入，不是原始RGB图像！**

```python
# VAE配置
patch_size = 2
in_channels = 12  # 3 * 2^2 = 12

# Encoder conv_in期望12个输入通道
WanCausalConv3d(12, 160, ...)
```

## 完整的数据流

### 原始流程（_encode方法）
```
RGB Image          Patchify         Encoder          Output
(1,3,F,512,512) -> (1,12,F,256,256) -> (1,48,Fl,32,32)
                   patch_size=2       8x spatial
                                      compression
```

### 关键点
1. **Patchify**: `3 channels × 2×2 = 12 channels`, `512×512 → 256×256`
2. **Encoder**: `256 → 128 → 64 → 32` (8x spatial compression)
3. **总压缩**: `512 → 32 = 16x` (patchify 2x + encoder 8x)

## 修正方案

### 1. compile_encoder.py修改

```python
# 错误 ❌
encoder_input = torch.rand((1, 3, 2, 512, 512))

# 正确 ✅
patch_size = vae.config.patch_size  # 2
in_channels = vae.config.in_channels  # 12
patchified_height = 512 // patch_size  # 256
patchified_width = 512 // patch_size  # 256

encoder_input = torch.rand((1, 12, 2, 256, 256))
```

### 2. feat_cache形状修改

```python
# 错误 ❌ - 使用原始分辨率
feat_cache = [
    torch.zeros((1, 12, 2, 512, 512)),   # conv_in
    torch.zeros((1, 160, 2, 512, 512)),  # down_blocks.0
    ...
]

# 正确 ✅ - 使用patchify后的分辨率
feat_cache = [
    torch.zeros((1, 12, 2, 256, 256)),   # conv_in
    torch.zeros((1, 160, 2, 256, 256)),  # down_blocks.0
    torch.zeros((1, 320, 2, 128, 128)),  # down_blocks.1
    torch.zeros((1, 640, 2, 64, 64)),    # down_blocks.2
    torch.zeros((1, 640, 2, 32, 32)),    # down_blocks.3 + mid_block
    ...
]
```

### 3. EncoderWrapper修改

```python
# _init_feat_cache_shapes现在正确处理patchify后的尺寸
def _init_feat_cache_shapes(self, x):
    # x is AFTER patchify: (batch, 12, frames, 256, 256)
    patchified_height = x.shape[3]  # 256
    patchified_width = x.shape[4]   # 256

    self.feat_cache_shapes = [
        (batch_size, 12, 2, patchified_height, patchified_width),  # 256x256
        ...
    ]
```

## 编译配置

### 512x512输入
```bash
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512
```

**实际编译：**
- Encoder输入: `(1, 12, 2, 256, 256)` ← patchify后
- feat_cache: 26层，从256x256开始下采样

### 256x256输入
```bash
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 256 \
    --width 256
```

**实际编译：**
- Encoder输入: `(1, 12, 2, 128, 128)` ← patchify后
- feat_cache: 26层，从128x128开始下采样

## 运行时行为

### I2V推理流程
```python
# 1. 用户提供原始图像
image = Image.open("cat.jpg").resize((512, 512))
# PIL Image: 512x512

# 2. 转换为tensor
video_input = transforms.ToTensor()(image)
# (3, 512, 512)

# 3. _encode内部处理
if vae.config.patch_size is not None:
    x = patchify(x, patch_size=2)
    # (1, 12, frames, 256, 256)

# 4. 调用编译后的encoder
vae.encoder(x, feat_cache=...)
# 输入: (1, 12, frames, 256, 256)
# 输出: (1, 48, latent_frames, 32, 32)
```

### EncoderWrapper处理
```python
# EncoderWrapper.forward接收patchify后的输入
def forward(self, x, **kwargs):
    # x.shape = (1, 12, 1, 256, 256) for single frame

    if original_frame_count == 1:
        # Duplicate to 2 frames for CACHE_T=2
        x = torch.cat([x, x], dim=2)
        # x.shape = (1, 12, 2, 256, 256)

    # Initialize feat_cache_shapes based on patchified size
    if self.feat_cache_shapes is None:
        self._init_feat_cache_shapes(x)
        # Uses x.shape[3] = 256, x.shape[4] = 256

    # Replace None with zero tensors
    feat_cache_fixed = [...]

    # Forward through compiled encoder
    output = self.model(x, feat_cache_fixed)
```

## 关键区别

| 项目 | 原始错误 | 修正后 |
|------|---------|--------|
| **输入channels** | 3 (RGB) | 12 (patchified) |
| **输入height** | 512 | 256 |
| **输入width** | 512 | 256 |
| **feat_cache第一层** | (1,12,2,512,512) | (1,12,2,256,256) |
| **最终latent size** | 64x64 ❌ | 32x32 ✅ |

## 验证

```python
# 检查patchify配置
vae.config.patch_size  # 2
vae.config.in_channels  # 12

# 检查encoder输入要求
vae.encoder.conv_in  # WanCausalConv3d(12, 160, ...)

# 完整测试
test_video = torch.randn(1, 3, 5, 512, 512)  # RGB输入
output = vae.encode(test_video).latent_dist.sample()
# output.shape = (1, 48, 2, 32, 32)  # 32x32 latent
#                                       ^^^^^^ 正确！
```

## 总结

✅ **修复完成的内容:**
1. compile_encoder.py使用patchify后的尺寸
2. feat_cache使用正确的空间分辨率（256→128→64→32）
3. EncoderWrapper正确处理patchify后的输入

✅ **现在可以编译:**
```bash
python neuron_wan2_2_ti2v/compile_encoder.py \
    --compiled_models_dir compile_workdir_latency_optimized \
    --height 512 \
    --width 512
```

**预期结果:**
- 编译成功
- 生成 `compile_workdir_latency_optimized/encoder/model.pt`
- 可用于I2V推理
