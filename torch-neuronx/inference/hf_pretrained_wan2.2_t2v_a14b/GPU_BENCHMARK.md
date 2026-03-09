# Wan2.2-T2V-A14B GPU Baseline Benchmark

GPU inference benchmark for [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) using diffusers, for performance comparison with Trainium2.

## Quick Start

```bash
pip install diffusers transformers accelerate

# 480P (official: 326.9s on H100)
python run_wan2.2_t2v_a14b_gpu.py --resolution 480P

# 720P (official: 1041.5s on H100)
python run_wan2.2_t2v_a14b_gpu.py --resolution 720P

# With CPU offload (reduces VRAM, needed for <80GB GPUs)
python run_wan2.2_t2v_a14b_gpu.py --resolution 480P --offload
```

## Parameters

Aligned with the [official Wan2.2 repo](https://github.com/Wan-Video/Wan2.2) and [HuggingFace model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers):

| Parameter | Value |
|-----------|-------|
| guidance_scale | 4.0 (high-noise expert) |
| guidance_scale_2 | 3.0 (low-noise expert) |
| num_inference_steps | 40 |
| num_frames | 81 |
| fps | 16 |
| max_sequence_length | 512 |

## Supported Resolutions

| Name | Size (H×W) | Aspect Ratio |
|------|-----------|--------------|
| 480P | 480 × 832 | 16:9 |
| 720P | 720 × 1280 | 16:9 |

## H100 Benchmark Results

All results on single H100 80GB, CUDA 13.0, PyTorch 2.9.1+cu130.

### With offload (`--offload`, comparable to official)

| Resolution | Official (FA3) | SDPA | FA4 |
|------------|---------------|------|-----|
| 480P | 326.9s / 41.3GB | 342.8s (8.06s/step) | **322.7s (7.56s/step)** |
| 720P | 1041.5s / 59.8GB | 1118.4s (27.33s/step) | **1010.6s (24.63s/step)** |

### Without offload (all models on GPU)

| Resolution | FA4 | vs Official (offload) |
|------------|-----|-----------------------|
| 480P | **277.7s (6.85s/step)** | **-15.0%** |

### Impact of offload (FA4, 480P)

| Config | Time | Per-step | Overhead |
|--------|------|----------|----------|
| FA4 no offload | 277.7s | 6.85s/step | — |
| FA4 + offload | 322.7s | 7.56s/step | **+16.2%** |

Offload adds ~0.71s per denoising step due to diffusers' CPU↔GPU model transfer hooks.

- **Official**: from [Wan2.2 repo](https://github.com/Wan-Video/Wan2.2) `comp_effic.png`, single H100, `--offload_model True --convert_model_dtype`, FlashAttention3
- **SDPA**: PyTorch `scaled_dot_product_attention` (no flash-attn installed), with `--offload`
- **FA4**: flash-attn-4 4.0.0b4 (CuTeDSL), `DIFFUSERS_ATTN_BACKEND="_flash_4"`

## Official Wan2.2 Performance Table (All GPUs)

From [Wan2.2 repo](https://github.com/Wan-Video/Wan2.2), format: Time (s) / Peak Memory (GB).

| GPU | Model | Resolution | 1 GPU | 4 GPUs | 8 GPUs |
|-----|-------|-----------|-------|--------|--------|
| 4090 | TI2V-5B T2V | 720P | 534.7/22.9 | 231.3/22.6 | 157.2/22.6 |
| | TI2V-5B I2V | 720P | 524.8/22.8 | 227.3/22.6 | 160.1/22.6 |
| H20 | T2V-A14B | 480P | 1133.9/41.3 | 306.6/40.9 | 170.5/26.2 |
| | | 720P | 4048.7/59.8 | 1067.8/51.7 | 564.7/37.8 |
| | I2V-A14B | 480P | 1117.4/41.0 | 305.4/40.8 | 173.4/26.1 |
| | | 720P | 4054.7/59.7 | 1076.9/51.6 | 577.0/37.0 |
| A100/A800 | T2V-A14B | 480P | 785.7/41.3 | 215.2/40.9 | 119.2/40.3 |
| | | 720P | 2735.7/59.8 | 725.3/51.5 | 386.7/37.6 |
| | I2V-A14B | 480P | 810.0/41.0 | 215.4/40.6 | 121.6/26.6 |
| | | 720P | 2810.9/59.7 | 730.5/51.6 | 393.4/37.0 |
| **H100/H800** | **T2V-A14B** | **480P** | **326.9/41.3** | 91.7/40.5 | 51.5/26.3 |
| | | **720P** | **1041.5/59.8** | 288.7/51.7 | 155.1/37.1 |
| | I2V-A14B | 480P | 327.8/41.0 | 92.4/40.8 | 52.9/26.3 |
| | | 720P | 1055.9/59.7 | 290.4/51.6 | 159.0/37.0 |

## Flash-Attn-4 Support (CUDA 13 + Hopper)

On CUDA 13, the standard `flash-attn` v2 package has no pre-built wheels. `flash-attn-4` uses CuTeDSL for runtime JIT kernel compilation, making it compatible with CUDA 13 without building from source.

### Setup

```bash
# 1. Install flash-attn-4
pip install --pre flash-attn-4

# 2. Patch diffusers to add _flash_4 backend
python patch_diffusers_fa4.py

# 3. Verify patch status
python patch_diffusers_fa4.py --check

# 4. Run with flash-attn-4
DIFFUSERS_ATTN_BACKEND="_flash_4" python run_wan2.2_t2v_a14b_gpu.py --resolution 480P
```

### How the Patch Works

`patch_diffusers_fa4.py` modifies 3 diffusers files (path auto-detected):

| File | Change |
|------|--------|
| `utils/import_utils.py` | Detect `flash_attn.cute` availability |
| `utils/__init__.py` | Export `is_flash_attn_4_available` function |
| `models/attention_dispatch.py` | Register `_flash_4` attention backend |

The patch is idempotent — running it twice is safe. Use `--check` to verify status.

### Notes

- First run triggers JIT kernel compilation (adds a few minutes to warmup). Subsequent runs use cached kernels.
- flash-attn-4 is currently in beta (4.0.0b4). Only supports Hopper (H100/H200) and Blackwell GPUs.
- The `_flash_4` backend does not support context parallel yet.
- diffusers' Wan model uses a pluggable backend dispatch (`dispatch_attention_fn`), so no changes to `transformer_wan.py` are needed.
