# Qwen-Image-Edit-2509 on AWS Trainium2

This project enables running the [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) model on AWS Trainium2 (trn2) instances using the Neuron SDK.

## Overview

Qwen-Image-Edit is a powerful image editing model that combines:
- **Text Encoder**: Qwen2.5-VL (Vision-Language Model)
  - Vision Encoder: 32 transformer blocks (~1.4GB)
  - Language Model: 28 layers with GQA (~7B params)
- **Transformer**: QwenImageTransformer2DModel for diffusion (~20.4B params)
- **VAE**: 3D convolutional autoencoder for image encoding/decoding

## Requirements

- AWS Trainium2 instance (trn2.48xlarge recommended)
- Neuron SDK 2.x with PyTorch support
- Python 3.10+

```bash
# Activate the Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## Environment Setup

### Mount NVMe Storage (trn2.48xlarge)

trn2.48xlarge 实例配有 4 个 NVMe 设备，需要挂载后才能使用。模型文件和编译产物较大，建议挂载到 `/opt/dlami/nvme`：

```bash
# 运行挂载脚本（需要 sudo 权限）
sudo ./setup_nvme.sh
```

脚本功能：
- 自动检测所有可用的 NVMe 设备（排除系统盘）
- 创建 RAID0 阵列以获得最大性能和容量（约 7.5TB）
- 格式化为 ext4 文件系统
- 挂载到 `/opt/dlami/nvme`
- 设置正确的权限（ubuntu 用户可读写）

**注意**：重启后需要重新运行脚本，或手动添加到 `/etc/fstab` 实现开机自动挂载。

## Quick Start

### 1. Download the Model

```bash
python neuron_qwen_image_edit/cache_hf_model.py
```

### 2. Apply Diffusers Patch (Required)

The original diffusers pipeline uses fixed 1024x1024 dimensions for VAE processing, which causes shape mismatches with our compiled model. Apply this patch:

```bash
# Location of the pipeline file
PIPELINE_FILE="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py"

# Backup original
cp $PIPELINE_FILE ${PIPELINE_FILE}.bak

# Apply patch: Change VAE_IMAGE_SIZE from fixed 1024*1024 to None
sed -i 's/^VAE_IMAGE_SIZE = 1024 \* 1024$/# VAE_IMAGE_SIZE = 1024 * 1024  # Original - causes patch mismatch on Neuron\nVAE_IMAGE_SIZE = None  # Will be set from height\/width in __call__/' $PIPELINE_FILE

# Also modify the vae_image_sizes computation (around line 687)
# Change: vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, ...)
# To:     vae_width, vae_height = calculate_dimensions(effective_vae_size, ...)
# And add before the loop:
#     effective_vae_size = height * width if VAE_IMAGE_SIZE is None else VAE_IMAGE_SIZE
```

**Why is this needed?**
The original pipeline processes source images at 1024x1024 regardless of target dimensions. This creates a shape mismatch:
- Target: 512x512 → 32x32 patches → 1024 patches
- Source: 1024x1024 → 64x64 patches → 4096 patches
- Total: 5120 patches (but model compiled for 2048)

After the patch:
- Target: 512x512 → 32x32 patches → 1024 patches
- Source: 512x512 → 32x32 patches → 1024 patches
- Total: 2048 patches (matches compiled model)

### 3. Compile Models

```bash
./compile.sh
```

Default compilation settings:
- Image size: 512x512
- Max sequence length: 512
- TP degree: 8
- Patch multiplier: 2 (for image editing)

### 4. Run Inference

```bash
# Basic usage (no CFG)
python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --transformer_batch_size 1

# With classifier-free guidance (requires batch_size=2 compilation)
python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --transformer_batch_size 2 \
    --guidance_scale 7.5

# Multi-image input (1-3 images supported)
python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo" \
    --transformer_batch_size 1
```

## Project Structure

```
hf_pretrained_qwen_image_edit/
├── README.md                      # This file
├── setup_nvme.sh                  # NVMe RAID0 mount script
├── compile.sh                     # Main compilation script
├── run_qwen_image_edit.py         # Inference script
├── neuron_qwen_image_edit/
│   ├── cache_hf_model.py          # Download model from HuggingFace
│   ├── compile_vae.py             # VAE compilation
│   ├── compile_transformer.py     # Transformer compilation (TP=8)
│   ├── compile_text_encoder.py    # Text encoder compilation
│   ├── neuron_commons.py          # Common utilities and wrappers
│   ├── neuron_parallel_utils.py   # Tensor parallelism utilities
│   ├── neuron_rope.py             # Neuron-compatible RoPE implementation
│   └── autoencoder_kl_qwenimage_neuron.py  # Neuron-compatible VAE
├── tests/                         # Unit tests for debugging
│   ├── run_all_tests.py           # Run all tests
│   ├── test_vae.py                # VAE encoder/decoder tests
│   ├── test_transformer.py        # Transformer tests
│   ├── test_text_encoder.py       # Text encoder tests
│   └── visualize_vae_diff.py      # VAE visual comparison
├── test_outputs/                  # Test output images (auto-created)
└── compiled_models/               # Output directory for compiled models
    ├── vae_encoder/
    ├── vae_decoder/
    ├── transformer/               # TP=8 sharded transformer (~5GB per shard)
    ├── vision_encoder/            # Single device
    └── language_model/            # TP=8 sharded
```

## Technical Details

### Model Size Analysis

| Component | Total Params | Sharded Layers | Per-Shard Size (TP=8) |
|-----------|-------------|----------------|----------------------|
| Transformer | 20.43B | attention, mlp, modulation | ~5.2 GB |
| Language Model | 7.07B | attention (GQA), mlp | ~1.8 GB |
| Vision Encoder | ~1.4B | N/A (single device) | ~2.8 GB |
| VAE | ~300M | N/A (DP=8) | ~600 MB |

### Tensor Parallelism Configuration

| Component | Parallelism | Notes |
|-----------|-------------|-------|
| Transformer | TP=8 | Full sharding including modulation layers |
| Language Model | TP=8 | KV head replication (4 heads → 8 shards) |
| Vision Encoder | Single device | Dimensions (3420) not divisible by 8 |
| VAE | DP=8 | Data parallel across 8 devices |

### Key Technical Challenges & Solutions

#### 1. Modulation Layer Sharding (Critical!)

The transformer has `img_mod` and `txt_mod` modulation layers that were previously NOT sharded, causing 6.8B params to be duplicated on every TP rank!

```python
# Problem: Each block has modulation layers [18432, 3072]
# 60 blocks × 2 mods × 56.6M params = 6.8B params duplicated!

# Solution: Shard modulation layers (neuron_parallel_utils.py)
def shard_modulation(mod: nn.Sequential) -> nn.Sequential:
    """Shard img_mod/txt_mod with ColumnParallelLinear."""
    orig_linear = mod[1]  # Sequential(SiLU, Linear)
    mod[1] = ColumnParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        gather_output=True,  # Need full output for modulation
        dtype=torch.bfloat16)
    ...
```

**Impact**: Reduces transformer size from ~17GB to ~5.2GB per shard!

#### 2. KV Head Replication for GQA

The Language Model uses Grouped Query Attention with only 4 KV heads, but we need TP=8. Solution: replicate KV heads across TP ranks.

```python
# In neuron_parallel_utils.py
def get_sharded_data_with_replication(data, dim, num_heads, tp_degree):
    """Shard data with head replication when num_heads < tp_degree."""
    if num_heads >= tp_size:
        return get_sharded_data(data, dim)
    else:
        replication_factor = tp_size // num_heads
        original_head_idx = tp_rank // replication_factor
        # Return the same head data for multiple TP ranks
        ...
```

#### 3. Q Head Padding

The model has 28 attention heads which isn't divisible by 8. Solution: pad to 32 heads.

```python
# Pad Q projection from 28 to 32 heads
padded_num_heads = ((num_heads + tp_degree - 1) // tp_degree) * tp_degree
```

#### 4. Image Editing Patch Multiplier

For image editing, the pipeline concatenates source image latents with noise latents, doubling the patch count. This is handled via `patch_multiplier=2`.

```python
# compile_transformer.py
temporal_frames = args.patch_multiplier  # 2 for editing, 1 for generation
num_patches = temporal_frames * patch_h * patch_w  # 2 * 32 * 32 = 2048
```

#### 5. Neuron-Compatible VAE

The original VAE uses `F.interpolate` with `mode='nearest-exact'` which isn't supported by Neuron. Solution: custom VAE with `mode='nearest'`.

```python
# autoencoder_kl_qwenimage_neuron.py
# Changed: mode="nearest-exact" → mode="nearest"
```

#### 6. Neuron-Compatible RoPE

The original RoPE implementation uses complex numbers which aren't supported. Solution: real-number implementation.

```python
# neuron_rope.py
def apply_rotary_pos_emb_neuron(x, cos, sin):
    """Apply rotary embeddings without complex numbers."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

#### 7. BFloat16 for Parallel Layers

The `neuronx_distributed` parallel layers (`ColumnParallelLinear`, `RowParallelLinear`) default to `float32`, which doubles the model size. Solution: explicitly set `dtype=torch.bfloat16`.

```python
# WRONG - defaults to float32, doubles model size!
attn.to_q = ColumnParallelLinear(in_features, out_features, ...)

# CORRECT - use bfloat16
attn.to_q = ColumnParallelLinear(dtype=torch.bfloat16, in_features, out_features, ...)
```

### Memory Optimization Summary

| Optimization | Before | After | Savings |
|-------------|--------|-------|---------|
| Modulation sharding | 17 GB/shard | 5.2 GB/shard | 70% |
| BFloat16 dtype | 2x size | 1x size | 50% |
| Proper weight sharding | Duplicated | 1/8 per rank | 87.5% |

## Compilation Options

```bash
./compile.sh [HEIGHT] [WIDTH] [IMAGE_SIZE] [TP_DEGREE] [MAX_SEQ_LEN] [PATCH_MULTIPLIER]

# Examples:
./compile.sh 512 512 224 8 512 2   # Default: 512x512, image editing
./compile.sh 1024 1024 224 8 512 2 # 1024x1024 output
./compile.sh 512 512 224 8 512 1   # Generation mode (no source image)
```

## Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--images` | Required | Input image path(s), 1-3 images |
| `--prompt` | Required | Edit instruction |
| `--height` | 512 | Output image height |
| `--width` | 512 | Output image width |
| `--compiled_height` | 512 | Height used during compilation |
| `--compiled_width` | 512 | Width used during compilation |
| `--num_inference_steps` | 50 | Denoising steps |
| `--guidance_scale` | 7.5 | CFG scale (requires batch_size=2) |
| `--transformer_batch_size` | 1 | 1=no CFG, 2=with CFG |
| `--max_sequence_length` | 512 | Must match compilation |
| `--patch_multiplier` | 2 | 2=editing, 1=generation |

## Troubleshooting

### Shape Mismatch Error
```
ERROR: hidden_states has X patches but model expects Y
```
**Solution**: Ensure the diffusers patch is applied and dimensions are consistent.

### OOM Error
```
Failed to allocate X bytes
```
**Solution**:
1. Ensure modulation layers are sharded (check for `shard_modulation` in compile_transformer.py)
2. Use `dtype=torch.bfloat16` for all parallel layers
3. Reduce image size or sequence length

### RoPE Dimension Mismatch
```
Shapes are not compatible: f32[1,2048,3,128] vs f32[1,1024,1,128]
```
**Solution**: Ensure `temporal_frames = patch_multiplier` in transformer compilation.

### TP World Size Conflict
```
Process group already initialized with different world_size
```
**Solution**: Both Transformer and Language Model must use the same TP degree (8).

### Vision Encoder TP Not Supported
```
3420 is not divisible by 8
```
**Solution**: Vision encoder uses single device (dimensions not compatible with TP=8).

## Unit Tests (Debugging Blurry Output)

如果推理输出的图片模糊，可以使用单元测试来定位问题。测试脚本会对比 Neuron 和 CPU 的推理结果差异。

### 测试文件结构

```
tests/
├── run_all_tests.py        # 运行所有测试
├── test_vae.py             # VAE encoder/decoder 测试
├── test_transformer.py     # Transformer 测试
├── test_text_encoder.py    # Text Encoder 测试
└── visualize_vae_diff.py   # VAE 输出可视化比较
```

### 运行所有测试

```bash
python tests/run_all_tests.py --compiled_models_dir /opt/dlami/nvme/compiled_models
```

### 单独测试各组件

```bash
# VAE 测试（推荐首先运行，最可能导致模糊）
python tests/test_vae.py --test all --save_images

# Transformer 测试
python tests/test_transformer.py --test single

# Text Encoder 测试
python tests/test_text_encoder.py --test all
```

### VAE 可视化比较（推荐用于调试模糊问题）

```bash
# 使用测试图像
python tests/visualize_vae_diff.py

# 使用自定义图像
python tests/visualize_vae_diff.py --input_image your_image.png
```

输出文件保存在 `test_outputs/` 目录：
- `vae_comparison.png` - 四格对比图（原图、CPU解码、Neuron解码、差异图）
- `vae_cpu_decoded.png` - CPU 解码结果
- `vae_neuron_decoded.png` - Neuron 解码结果
- `vae_diff_20x.png` - 差异放大 20 倍

### 测试指标说明

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| Cosine Similarity | 余弦相似度 | > 0.99 正常，< 0.95 有问题 |
| Max Absolute Error | 最大绝对误差 | < 0.1 正常 |
| Mean Absolute Error | 平均绝对误差 | < 0.01 正常 |

### 调试建议

如果图片模糊，按以下顺序排查：

1. **VAE Decoder**（最可能）
   - 运行 `visualize_vae_diff.py` 查看差异图
   - 检查 Cosine Similarity 是否 < 0.99
   - 常见问题：插值模式不兼容、数值精度损失

2. **Transformer**
   - 检查多个 timestep 的累积误差
   - 常见问题：RoPE 编码、注意力实现

3. **Text Encoder**
   - Vision encoder 误差影响条件生成
   - 常见问题：embedding 层、注意力层

4. **数值精度**
   - 检查 bfloat16/float32 转换
   - 检查 latent_mean/latent_std 缩放

## Known Limitations

1. **Fixed dimensions**: Models are compiled for specific dimensions. Different sizes require recompilation.
2. **Vision encoder**: Must run on single device (hidden dim 3420 not divisible by 8).
3. **Sequence length**: Must match between compilation and inference.

## References

- [Qwen-Image-Edit Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## License

This code is provided for research and educational purposes. Please refer to the original model license for usage terms.
