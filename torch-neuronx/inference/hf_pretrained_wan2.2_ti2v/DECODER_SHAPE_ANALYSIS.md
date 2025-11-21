# Decoder 形状分析报告

## 输入参数
- **height**: 512
- **width**: 512
- **num_frames**: 15

## Latent 空间维度计算

```python
latent_height = height // 16 = 512 // 16 = 32
latent_width = width // 16 = 512 // 16 = 32
latent_frames = (num_frames - 1) // 4 + 1 = (15 - 1) // 4 + 1 = 4
```

**为什么除以 16？**
- VAE 将空间维度降低 8 倍
- Transformer 的 patch_embedding 再降低 2 倍
- 总共: 8 × 2 = 16

## Decoder 各层形状详细列表

### 输入层
| 层名称 | 形状 | 内存占用 |
|--------|------|----------|
| Decoder 输入 | `(1, 48, 4, 32, 32)` | 0.75 MB |

### Feat_cache 各层 (CACHE_T=2)

#### Mid Block
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 0 | conv_in | `(1, 48, 2, 32, 32)` | 0.38 MB |
| 1 | mid_block.resnets.0.conv1 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 2 | mid_block.resnets.0.conv2 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 3 | mid_block.resnets.1.conv1 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 4 | mid_block.resnets.1.conv2 | `(1, 1024, 2, 32, 32)` | 8.00 MB |

#### Up Block 0 (32×32 分辨率)
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 5 | up_blocks.0.resnets.0.conv1 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 6 | up_blocks.0.resnets.0.conv2 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 7 | up_blocks.0.resnets.1.conv1 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 8 | up_blocks.0.resnets.1.conv2 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 9 | up_blocks.0.resnets.2.conv1 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 10 | up_blocks.0.resnets.2.conv2 | `(1, 1024, 2, 32, 32)` | 8.00 MB |
| 11 | up_blocks.0.upsampler.time_conv | `(1, 1024, 2, 32, 32)` | 8.00 MB |

#### Up Block 1 (64×64 分辨率)
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 12 | up_blocks.1.resnets.0.conv1 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 13 | up_blocks.1.resnets.0.conv2 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 14 | up_blocks.1.resnets.1.conv1 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 15 | up_blocks.1.resnets.1.conv2 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 16 | up_blocks.1.resnets.2.conv1 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 17 | up_blocks.1.resnets.2.conv2 | `(1, 1024, 2, 64, 64)` | 32.00 MB |
| 18 | up_blocks.1.upsampler.time_conv | `(1, 1024, 2, 64, 64)` | 32.00 MB |

#### Up Block 2 (128×128 → 256×256 分辨率)
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 19 | up_blocks.2.resnets.0.conv1 | `(1, 1024, 2, 128, 128)` | **128.00 MB** |
| 20 | up_blocks.2.resnets.0.conv2 | `(1, 512, 2, 128, 128)` | 64.00 MB |
| 21 | up_blocks.2.resnets.0.conv_shortcut | `(1, 512, 2, 128, 128)` | 64.00 MB |
| 22 | up_blocks.2.resnets.1.conv1 | `(1, 512, 2, 128, 128)` | 64.00 MB |
| 23 | up_blocks.2.resnets.1.conv2 | `(1, 512, 2, 128, 128)` | 64.00 MB |
| 24 | up_blocks.2.resnets.2.conv1 | `(1, 512, 2, 128, 128)` | 64.00 MB |
| 25 | up_blocks.2.resnets.2.conv2 | `(1, 512, 2, 256, 256)` | **256.00 MB** ⚠️ 最大 |

#### Up Block 3 (256×256 分辨率)
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 26 | up_blocks.3.resnets.0.conv1 | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 27 | up_blocks.3.resnets.0.conv2 | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 28 | up_blocks.3.resnets.0.conv_shortcut | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 29 | up_blocks.3.resnets.1.conv1 | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 30 | up_blocks.3.resnets.1.conv2 | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 31 | up_blocks.3.resnets.2.conv1 | `(1, 256, 2, 256, 256)` | **128.00 MB** |
| 32 | up_blocks.3.resnets.2.conv2 | `(1, 256, 2, 256, 256)` | **128.00 MB** |

#### 输出层
| 索引 | 层名称 | 形状 | 内存占用 |
|------|--------|------|----------|
| 33 | conv_out | `(1, 12, 2, 256, 256)` | 6.00 MB |

### 输出层
| 层名称 | 形状 | 内存占用 |
|--------|------|----------|
| Decoder 输出 | `(1, 12, 4, 256, 256)` | 12.00 MB |

## 内存占用总结

| 组件 | 内存占用 | 占比 |
|------|----------|------|
| Decoder 输入 | 0.75 MB | 0.04% |
| Feat_cache (34 layers) | **1918.38 MB** | **99.3%** |
| Decoder 输出 | 12.00 MB | 0.6% |
| **总计** | **1931.12 MB (1.89 GB)** | 100% |

## 内存瓶颈分析

### Top 10 内存占用最大的层

1. **up_blocks.2.resnets.2.conv2** (索引 25): 256.00 MB - `(1, 512, 2, 256, 256)`
2. **up_blocks.2.resnets.0.conv1** (索引 19): 128.00 MB - `(1, 1024, 2, 128, 128)`
3. **up_blocks.3 的 7 个层** (索引 26-32): 各 128.00 MB - `(1, 256, 2, 256, 256)`

### 关键发现

1. **feat_cache 是主要瓶颈**
   - 占总内存的 99.3%
   - 固定大小 (1918.38 MB)，**不随 num_frames 变化**
   - 因为 CACHE_T=2 (缓存帧数固定)

2. **后期 up_blocks 占用最多**
   - up_blocks.2 和 up_blocks.3 因为空间分辨率大 (128×128, 256×256)
   - 单个层最大可达 256 MB

3. **num_frames 的影响很小**
   - 从 num_frames=7 增加到 15，总内存仅增加 6.38 MB
   - 因为 feat_cache 不变，只有输入/输出增大

## 不同 num_frames 配置对比

| num_frames | latent_frames | Decoder 输入 | Feat_cache | Decoder 输出 | 总计 |
|------------|---------------|--------------|------------|--------------|------|
| 4 | 1 | 0.19 MB | 1918.38 MB | 3.00 MB | 1921.56 MB (1.88 GB) |
| 7 | 2 | 0.38 MB | 1918.38 MB | 6.00 MB | 1924.75 MB (1.88 GB) |
| **15** | **4** | **0.75 MB** | **1918.38 MB** | **12.00 MB** | **1931.12 MB (1.89 GB)** |
| 31 | 8 | 1.50 MB | 1918.38 MB | 24.00 MB | 1943.88 MB (1.90 GB) |

## 优化建议

### 1. 使用低精度 (最有效)

当前 decoder 使用 **float32**，如果改用 **bfloat16**:
- feat_cache: 1918.38 MB → **959.19 MB** (减少 50%)
- 总内存: 1931.12 MB → **965.56 MB** (~0.94 GB，减少 50%)

**实施方法**:
```python
# 在 compile_decoder.py 中修改
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae",
    torch_dtype=torch.bfloat16,  # 改为 bfloat16
    cache_dir="wan2.2_ti2v_hf_cache_dir"
)
```

### 2. 降低空间分辨率

- 从 512×512 降到 256×256:
  - latent: 32×32 → 16×16
  - feat_cache 将减少到原来的 1/4
  - 但会影响输出视频质量

### 3. num_frames 不是瓶颈

- **从 7 增加到 15 只增加 6.38 MB**
- feat_cache 大小不受 num_frames 影响
- 可以安全地增加 num_frames 而不会显著增加内存

### 4. 分块处理 (如果需要更大的 num_frames)

如果需要更长的视频 (num_frames > 31):
- 可以分块处理，每次处理 15-31 帧
- 因为 feat_cache 固定，分块不会增加峰值内存

## 结论

1. **decoder 不是 num_frames 的瓶颈**
   - feat_cache 固定大小，不随帧数变化
   - 从 7 帧增加到 15 帧几乎没有内存增加

2. **真正的瓶颈是 feat_cache 的精度**
   - 使用 float32 导致约 1.9 GB 内存占用
   - 改用 bfloat16 可以减半

3. **建议优先尝试**
   - 将 decoder 改为 bfloat16 精度
   - 这样可以支持更大的 num_frames (甚至 31+)
