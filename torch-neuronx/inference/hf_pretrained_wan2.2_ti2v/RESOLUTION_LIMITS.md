# Wan2.2 分辨率限制说明

## 限制来源

### 1. **数学限制**（必须满足）

模型架构决定的硬性限制：

```
总下采样倍数 = VAE (8x) × Transformer patch (2x) = 16x
```

**要求：分辨率必须能被 16 整除**

#### 可用的分辨率
```
128, 256, 384, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024...
```

### 2. **Neuron 编译限制**（真正的限制）

#### 位置 1: `compile_transformer_latency_optimized.py`
```python
# 第 87-88 行
latent_height = args.height//16
latent_width = args.width//16

# 第 103 行 - 计算序列长度
seq_len = (latent_frames // patch_size_t) * (latent_height // patch_size_h) * (latent_width // patch_size_w)

# 第 105 行 - 创建示例输入
sample_hidden_states = torch.ones((batch_size, in_channels, latent_frames, latent_height, latent_width), ...)
```

#### 位置 2: `compile_decoder.py`
```python
# 第 40-41 行
latent_height = args.height//16
latent_width = args.width//16

# 第 65 行 - decoder 输入
decoder_input = torch.rand((batch_size, in_channels, decoder_frames, latent_height, latent_width), ...)

# 第 69-102 行 - feat_cache 形状
feat_cache = [
    torch.rand((batch_size, 48, 2, latent_height, latent_width), ...),    # 32x32
    torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), ...),  # 64x64
    torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), ...),   # 256x256
    ...
]

# 第 120 行 - post_quant_conv 输入
post_quant_conv_input = torch.rand((batch_size, in_channels, latent_frames, latent_height, latent_width), ...)
```

#### 位置 3: `compile_latency_optimized.sh`
```bash
# 第 16-20 行 - transformer 编译
python neuron_wan2_2_ti2v/compile_transformer_latency_optimized.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--max_sequence_length 512 \
--height 512 \    # ← 这里固定分辨率
--width 512       # ← 这里固定分辨率

# 第 23-26 行 - decoder 编译
python neuron_wan2_2_ti2v/compile_decoder.py \
--compiled_models_dir "compile_workdir_latency_optimized" \
--height 512 \    # ← 这里固定分辨率
--width 512       # ← 这里固定分辨率
```

### 3. **运行时限制**

#### 位置 4: `run_wan2.2_ti2v_latency_optimized.py`
```python
# 第 89-90 行
output_warmup = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,  # ← 必须与编译时一致
    width=512,   # ← 必须与编译时一致
    num_frames=15,
    ...
)
```

## 为什么只能用编译时的分辨率？

**Neuron 编译器将模型编译为固定形状的二进制**：

1. **Transformer 编译时**：
   - 输入 tensor 形状：`(1, 48, latent_frames, latent_height, latent_width)`
   - 序列长度固定：`seq_len = latent_frames * latent_height * latent_width // 4`

2. **Decoder 编译时**：
   - 输入 tensor 形状：`(1, 48, 2, latent_height, latent_width)`
   - feat_cache 的 34 个 tensor 形状都固定

3. **运行时**：
   - 传入不同形状的 tensor 会报错：`Incorrect tensor shape`

## 如何使用其他分辨率？

### 步骤 1: 修改编译脚本
编辑 `compile_latency_optimized.sh`:
```bash
# 例如改为 768x768
--height 768 \
--width 768
```

### 步骤 2: 重新编译
```bash
bash compile_latency_optimized.sh
```

需要重新编译的组件：
- ✓ Text encoder（不受影响，但建议重新编译）
- ✓ **Transformer**（必须重新编译）
- ✓ **Decoder**（必须重新编译）
- ✓ **post_quant_conv**（必须重新编译）

### 步骤 3: 修改运行脚本
编辑 `run_wan2.2_ti2v_latency_optimized.py`:
```python
output = pipe(
    prompt=prompt,
    height=768,  # 改为新分辨率
    width=768,   # 改为新分辨率
    ...
)
```

## 推荐的分辨率

| 分辨率 | Latent 尺寸 | 序列长度 (4帧) | 内存占用 | 备注 |
|--------|-------------|----------------|----------|------|
| 256×256 | 16×16 | 256 | 低 | 快速测试 |
| 384×384 | 24×24 | 576 | 中低 | |
| 480×480 | 30×30 | 900 | 中 | |
| **512×512** | **32×32** | **1024** | **中** | **默认/常用** |
| 640×640 | 40×40 | 1600 | 中高 | |
| 768×768 | 48×48 | 2304 | 高 | 高质量 |
| 1024×1024 | 64×64 | 4096 | 很高 | 最高质量 |

## 注意事项

1. **分辨率必须能被 16 整除**
   - ✓ 512, 768, 1024
   - ✗ 500, 700, 1000

2. **编译时间会随分辨率增加**
   - 分辨率越大，序列长度越长
   - Transformer 编译时间 ∝ seq_len²

3. **内存占用会显著增加**
   - Decoder feat_cache: `1918 MB × (resolution/512)²`
   - 例如 768×768: ~4.3 GB feat_cache

4. **可以为不同分辨率创建不同的编译版本**
   ```bash
   # 512×512 版本
   --compiled_models_dir "compile_workdir_512"

   # 768×768 版本
   --compiled_models_dir "compile_workdir_768"
   ```

## 总结

**限制根源**：Neuron 编译器将模型编译为固定输入形状的二进制，无法动态改变。

**解决方案**：
1. 只使用一种分辨率 → 编译一次
2. 需要多种分辨率 → 为每种分辨率编译一个版本
3. 频繁切换分辨率 → 不适合使用 Neuron（建议 GPU）
