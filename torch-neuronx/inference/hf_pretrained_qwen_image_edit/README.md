# Qwen-Image-Edit on AWS Trainium2

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

trn2.48xlarge instances come with 4 NVMe devices that need to be mounted. Model files and compilation artifacts are large, so mounting to `/opt/dlami/nvme` is recommended:

```bash
# Run the mount script (requires sudo)
sudo ./setup_nvme.sh
```

The script will:
- Auto-detect all available NVMe devices (excluding system disk)
- Create RAID0 array for maximum performance and capacity (~7.5TB)
- Format as ext4 filesystem
- Mount to `/opt/dlami/nvme`
- Set correct permissions (ubuntu user can read/write)

**Note**: After reboot, run the script again or add to `/etc/fstab` for auto-mount.

## Quick Start

### 1. Download the Model

```bash
python neuron_qwen_image_edit/cache_hf_model.py
```

### 2. Compile Models

```bash
./compile.sh
```

Default compilation settings:
- Output image size: 1024x1024
- VAE tile size: 512x512 (fixed, uses tiled processing)
- Max sequence length: 1024
- TP degree: 8
- Patch multiplier: 3 (for 2-image merging)

### 3. Run Inference

```bash
# Two-image merging (default, requires patch_multiplier=3)
python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo"

# Single image editing (requires patch_multiplier=2)
python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --patch_multiplier 2

# With custom CFG scale
python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --true_cfg_scale 6.0
```

**Note**: CFG (Classifier-Free Guidance) runs the transformer twice sequentially per step, not with batch_size=2.

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
├── tests/                         # Unit tests
└── compiled_models/               # Output directory for compiled models
    ├── vae_encoder/
    ├── vae_decoder/
    ├── transformer/               # TP=8 sharded transformer
    └── vision_encoder/            # Single device
```

## Technical Details

### Model Architecture and Execution

| Component | Total Params | Execution | Notes |
|-----------|-------------|-----------|-------|
| Transformer | 20.43B | TP=8 on Neuron | ~5.2 GB/shard |
| Language Model | 7.07B | CPU | GQA 28Q/4KV incompatible with TP=8 |
| Vision Encoder | ~1.4B | CPU (default) | Can use Neuron with `--neuron_vision_encoder` |
| VAE | ~300M | DP=8 on Neuron | Tiled processing for large images |

**Why Language Model runs on CPU**: The Qwen2.5-VL language model uses Grouped Query Attention with 28 Q heads and 4 KV heads (group size = 7). With TP=8, the Q-KV mapping cannot be preserved correctly. Valid TP degrees are only 1, 2, or 4.

### Key Technical Implementations

#### 1. Modulation Layer Sharding

The transformer has `img_mod` and `txt_mod` modulation layers. Without sharding, 6.8B params would be duplicated on every TP rank.

```python
# Each block has modulation layers [18432, 3072]
# 60 blocks × 2 mods × 56.6M params = 6.8B params

# Solution: Shard with ColumnParallelLinear (gather_output=True)
```

**Impact**: Reduces transformer size from ~17GB to ~5.2GB per shard.

#### 2. Neuron-Compatible VAE

The original VAE uses `F.interpolate` with `mode='nearest-exact'` which isn't supported by Neuron. Custom VAE uses `mode='nearest'`.

#### 3. Neuron-Compatible RoPE

The original RoPE implementation uses complex numbers which aren't supported. Custom implementation uses real-number operations with correct interleaved expansion.

#### 4. M-RoPE Position IDs

Qwen2.5-VL uses Multimodal RoPE requiring 3D position_ids `[3, batch, seq]`:
- Text tokens: sequential positions (same for t, h, w dimensions)
- Image tokens: 3D grid positions based on spatial layout

## Compilation Options

```bash
./compile.sh [HEIGHT] [WIDTH] [IMAGE_SIZE] [TP_DEGREE] [MAX_SEQ_LEN] [PATCH_MULTIPLIER]

# Examples:
./compile.sh 1024 1024 448 8 1024 3   # Default: 1024x1024, 2-image merge
./compile.sh 1024 1024 448 8 1024 2   # Single image editing
./compile.sh 512 512 224 8 512 2      # 512x512 output
```

## Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--images` | Required | Input image path(s), 1-3 images |
| `--prompt` | Required | Edit instruction |
| `--height` | 1024 | Output image height (must match compiled model) |
| `--width` | 1024 | Output image width (must match compiled model) |
| `--patch_multiplier` | 2 | 2=single image editing, 3=two-image merge |
| `--image_size` | 448 | Vision encoder input size |
| `--max_sequence_length` | 1024 | Text sequence length (must match compilation) |
| `--num_inference_steps` | 40 | Denoising steps |
| `--true_cfg_scale` | 4.0 | CFG scale (runs transformer twice per step) |
| `--cpu_vision_encoder` | True | Run vision encoder on CPU (default) |
| `--neuron_vision_encoder` | False | Run vision encoder on Neuron |
| `--vae_tile_size` | 512 | VAE tile size for tiled processing |

## Troubleshooting

### Shape Mismatch Error
```
ERROR: hidden_states has X patches but model expects Y
```
**Solution**: Ensure `--height`, `--width`, and `--patch_multiplier` match the dimensions used during compilation.

### OOM Error
```
Failed to allocate X bytes
```
**Solution**:
1. Ensure modulation layers are sharded
2. Use `dtype=torch.bfloat16` for all parallel layers
3. Reduce image size or sequence length

### RoPE Dimension Mismatch
```
Shapes are not compatible: f32[1,2048,3,128] vs f32[1,1024,1,128]
```
**Solution**: Ensure `patch_multiplier` matches between compilation and inference.

## Known Limitations

1. **Fixed dimensions**: Models are compiled for specific dimensions. Different sizes require recompilation.
2. **Language model**: Runs on CPU due to GQA architecture (28Q/4KV heads incompatible with TP=8).
3. **Sequence length**: Must match between compilation and inference.
4. **Performance**: Transformer is the bottleneck (~2.8s/step on TRN2 vs ~0.75s/step on H100 for 1024x1024 with 2-image merge).

## References

- [Qwen-Image-Edit Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## License

This code is provided for research and educational purposes. Please refer to the original model license for usage terms.
