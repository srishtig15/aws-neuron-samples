# Wan2.2 Text/Image-to-Video Inference on AWS Trainium2

This project implements [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) video generation on AWS Trainium2 (trn2.48xlarge) using the AWS Neuron SDK. Supports multiple resolutions from 512x384 up to 1280x704 (720P) with text-to-video and image-to-video generation.

## Multi-Resolution Performance (trn2.48xlarge vs H100)

| Resolution | FPS | Frames | Trn2 (s) | H100 (s) | Decoder |
|-----------|-----|--------|-----------|-----------|---------|
| 512x384 | 16 | 81 | 32.70 | 16.13 | rolling |
| 512x384 | 24 | 121 | 49.24 | 24.48 | rolling |
| 640x480 | 16 | 81 | 55.38 | 26.06 | rolling |
| 640x480 | 24 | 121 | 81.50 | 39.67 | rolling |
| 1280x704 | 16 | 81 | 163.88 | 87.66 | tiled |
| 1280x704 | 24 | 121 | 260.01 | 143.20 | tiled |

Timing is pure inference (excludes model loading and warmup). See `test_results.txt` and `test_results_gpu.txt`.

## Quick Start (Recommended: V3 CP)

```bash
# Activate Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install -r requirements.txt

# Compile all models (text encoder, transformer, decoder, encoder)
./compile_v3_cp.sh

# Text-to-Video (T2V)
python run_wan2.2_ti2v_v3_cp.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_cp \
    --prompt "A cat walks on the grass, realistic"

# Image-to-Video (I2V)
python run_wan2.2_ti2v_v3_cp.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_cp \
    --image input.png \
    --prompt "A cat walks on the grass, realistic"
```

## Architecture Overview

The Wan2.2 pipeline has 5 main components, each compiled separately for Neuron:

```
Text Prompt                Input Image (I2V only)
    |                           |
    v                           v
[Text Encoder]            [VAE Encoder]    bfloat16, patchify → encode
  UMT5, TP=4                   |
    |                     [quant_conv]     float32
    |                           |
    v                           v
[Transformer]   DiT-based diffusion, 50 denoising steps
    |            TP=4, Context Parallel (CP=2), world_size=8
    v            (I2V: frame 0 = image latent, frames 1-N = noise)
[post_quant_conv]  3D convolution, float32
    |
    v
[VAE Decoder]   Conv3D upsampling, bfloat16, 11 temporal chunks
    |
    v
Video Output (512x512, 81 frames)
```

### Performance Breakdown (V3 CP, trn2.48xlarge)

| Component | Time | Details |
|-----------|------|---------|
| Transformer | ~21s | 50 steps @ 0.43s/step |
| VAE Decoder | ~5.6s | 11 calls @ 0.50s/call (NoCache) |
| post_quant_conv | ~0.003s | Single call |
| **Total** | **~27s** | |

## Compilation Approaches

This project evolved through multiple optimization iterations. The recommended approach is **V3 CP**.

### V3 CP (Recommended)

- **Script**: `compile_v3_cp.sh` / `run_wan2.2_ti2v_v3_cp.py`
- **Transformer**: TP=4, CP=2 (Context Parallel), world_size=8
- **Decoder**: bfloat16, NoCache (feat_cache as zero buffers), `--model-type=unet-inference`
- **Key innovations**:
  - Context Parallel splits the sequence dimension across 2 groups, enabling each group to process half the tokens
  - `local_rms_norm` avoids Neuron compiler all-reduce bugs while preserving normalization accuracy
  - Decoder compiled with `--model-type=unet-inference` (not `transformer`) for Conv3D-optimized code generation
  - bfloat16 decoder halves memory bandwidth
  - NoCache decoder internalizes feat_cache as zero buffers, eliminating ~960MB/call data transfer (0.50s vs 0.78s per call)

```bash
./compile_v3_cp.sh [output_dir] [compiler_workdir]
# Default: /opt/dlami/nvme/compiled_models_v3_cp
```

### V3 Flash (Alternative)

- **Script**: `compile_v3_flash.sh` / `run_wan2.2_ti2v_v3_flash.py`
- **Transformer**: TP=8, NKI Flash Attention
- **Decoder**: bfloat16, `--model-type=unet-inference`
- Simpler parallelism (no Context Parallel), relies on NKI Flash Attention for the transformer

```bash
./compile_v3_flash.sh [output_dir] [compiler_workdir]
# Default: /opt/dlami/nvme/compiled_models_v3_flash
```

### V2 (Legacy)

- **Script**: `compile_latency_optimized_v2.sh` / `run_wan2.2_ti2v_latency_optimized_v2.py`
- **Transformer**: TP=8, standard attention
- **Decoder**: V1 TorchScript (single core)
- Uses ModelBuilder API for text encoder and transformer, falls back to V1 TorchScript for decoder

### V1 (Deprecated)

- **Script**: `compile_latency_optimized.sh` / `run_wan2.2_ti2v_latency_optimized.py`
- All TorchScript via deprecated `parallel_model_trace()` API

## Key Optimizations

### 1. Context Parallel (V3 CP Transformer)

The transformer sequence length is 5376 tokens (21 latent frames x 256 spatial). Context Parallel splits this across 2 groups:

- **TP=4**: Each tensor parallel rank handles 1/4 of attention heads
- **CP=2**: Each CP rank handles 1/2 of the sequence
- Self-attention uses scatter/gather collectives for cross-sequence communication
- Cross-attention (text conditioning) doesn't need CP since the text sequence is shared

Implementation: `neuron_wan2_2_ti2v/compile_transformer_v3_cp.py`

### 2. local_rms_norm (Compiler Bug Workaround)

The Neuron compiler generates incorrect all-reduce replica groups `[[0,1,2,3]]` for `DistributedRMSNorm`, which causes assertion errors at runtime with world_size=8 (expecting `[[0,1,2,3],[4,5,6,7]]`).

Solution: `local_rms_norm` computes RMSNorm locally on each rank's shard without any all-reduce:

```python
def local_rms_norm(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(x.dtype)
```

Applied to Q/K normalization in both self-attention and cross-attention. The difference from global norm is negligible for QK normalization since each rank's local hidden dimension (~1024) is large enough for stable statistics.

### 3. VAE Decoder: bfloat16 + NoCache

The VAE decoder is dominated by Conv3D operations. Two key optimizations:

**bfloat16**: Halves memory bandwidth (the bottleneck for Conv3D). Decoder time reduced from ~20s (V1 float32) to ~8.6s (V3 bfloat16).

**NoCache**: The decoder's `feat_cache` (34 tensors, ~960MB) is always zeros between calls because NxDModel copies inputs to device without reflecting device-side mutations back to host. Instead of transferring these zeros every call, they are registered as constant buffers inside the compiled model:
- Eliminates ~960MB CPU→Device transfer per decoder call
- Decoder per-call: ~0.50s (vs ~0.78s with external feat_cache)
- Total decoder time: ~5.6s (vs ~8.6s), a 35% speedup

The `DecoderWrapperV3NoCache` only passes `x` (~192KB) to the NxDModel, handling dtype conversion and temporal frame padding/trimming.

### 4. Correct Compiler Model Type

ModelBuilder defaults to `--model-type=transformer` which optimizes for attention patterns. The VAE decoder is Conv3D-heavy, so we explicitly pass `--model-type=unet-inference`:

```python
traced_decoder = decoder_builder.compile(
    compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
)
```

### 5. VAE Encoder V3 (Image-to-Video)

For I2V mode, the input image is encoded into latent space using the VAE encoder + quant_conv:

- **Encoder**: bfloat16, `--model-type=unet-inference` (Conv3D-heavy like the decoder)
- **quant_conv**: float32 (cheap, runs once, benefits from precision)
- Compiled WITHOUT `feat_cache` since I2V only encodes a single image
- Input is patchified (patch_size=2): `(1, 12, 2, 256, 256)` for 512x512
- Compiled with 2 temporal frames to avoid causal conv edge cases (1-frame input is padded)

The `EncoderWrapperV3` handles:
- Frame padding: 1 input frame → 2 frames for the compiled model
- dtype conversion: float32 → bfloat16 → float32
- Ignoring `feat_cache`/`feat_idx` arguments from the `_encode()` loop

Implementation: `neuron_wan2_2_ti2v/compile_encoder_v3.py`

### 6. Temporal Chunked Decoding

The VAE decoder processes latent frames in chunks of 2 (CACHE_T=2) with causal temporal caching (`feat_cache`). For 81 frames (21 latent frames):
- Call 1: First frame (with `first_chunk=True`)
- Calls 2-11: Two frames per call

This is implemented in the custom `autoencoder_kl_wan.py` which replaces the diffusers default.

### 7. Tiled Spatial Decode (720P+)

At 720P (1280x704), the VAE decoder's Conv3D operators exceed the Neuron compiler's per-operator instruction limit (`NCC_EXTP003`: 1.2M instructions vs 300K limit). This is different from the total NEFF instruction limit (`NCC_EVRF007`) which can be bypassed with `--tiled-inst-limit`.

**Solution**: Compile the decoder at a small tile resolution (e.g., 384x512), then tile the full-resolution latent at inference time with overlap blending.

**Key design points**:
- The VAE's `feat_cache` is purely temporal (dim=2, CACHE_T=2) with no spatial context, so spatial tiles are fully independent
- All Conv3D kernels use 3x3x3 with padding=1 (same-padding), so spatial tiling introduces no boundary artifacts beyond the overlap region
- Each tile maintains its own independent rolling cache (34 tensors per tile)
- Memory-efficient: processes all tiles for one temporal chunk before moving to the next chunk

**Tiling parameters** (for 1280x704 with 384x512 tiles):
- Latent space: 44x80 → tiled with 24x32 tiles, overlap=4 latent pixels
- Produces 3x3 = 9 tiles per temporal chunk
- Overlap regions use linear ramp blending weights

**Blending**: Each tile gets a 2D weight mask with linear ramps in overlap regions:
- Interior pixels: weight = 1.0
- Overlap pixels: linear ramp from 0.0 to 1.0
- Image boundary pixels: weight = 1.0 (no ramp at edges)

Implementation: `DecoderWrapperV3Tiled` in `neuron_wan2_2_ti2v/neuron_commons.py`

## File Structure

### Core Compilation Scripts (`neuron_wan2_2_ti2v/`)

| File | Description |
|------|-------------|
| `compile_transformer_v3_cp.py` | V3 CP transformer (TP=4, CP=2, local_rms_norm) |
| `compile_transformer_v3_flash.py` | V3 Flash transformer (TP=8, NKI Flash Attention) |
| `compile_text_encoder_v2.py` | Text encoder (ModelBuilder API) |
| `compile_decoder_v3_nocache.py` | VAE decoder (bfloat16, NoCache, `--model-type=unet-inference`) |
| `compile_decoder_v3_rolling.py` | VAE decoder with rolling cache (non-TP, for tiled decode) |
| `compile_decoder_v3.py` | VAE decoder with external feat_cache (legacy) |
| `compile_encoder_v3.py` | VAE encoder + quant_conv (for I2V) |
| `cache_hf_model.py` | Download and cache HuggingFace model |

### Runtime

| File | Description |
|------|-------------|
| `run_wan2.2_ti2v_v3_cp.py` | V3 CP inference script |
| `run_wan2.2_ti2v_v3_flash.py` | V3 Flash inference script |
| `autoencoder_kl_wan.py` | Custom VAE with temporal chunked decoding |

### Wrappers and Utilities (`neuron_wan2_2_ti2v/`)

| File | Description |
|------|-------------|
| `neuron_commons.py` | Runtime wrappers: `DecoderWrapperV3NoCache`, `DecoderWrapperV3Tiled`, `DecoderWrapperV3`, `EncoderWrapperV3`, `QuantConvWrapperV3`, attention utilities |
| `neuron_commons_v2.py` | `InferenceTextEncoderWrapperV2` for NxDModel text encoder |
| `neuron_parallel_utils.py` | Tensor parallel utilities for UMT5 sharding |
| `distributed_rmsnorm.py` | Distributed RMSNorm (reference, not used in V3 due to compiler bug) |

### Shell Scripts

| File | Description |
|------|-------------|
| `compile_v3_cp.sh` | **Recommended**: Full V3 CP compilation pipeline |
| `compile_v3_flash.sh` | V3 Flash compilation pipeline |
| `compile_latency_optimized_v2.sh` | V2 compilation pipeline (legacy) |
| `test_resolutions.sh` | Multi-resolution test suite (auto-tiling for 720P+) |

## Inference Options

### Text-to-Video (T2V)

```bash
python run_wan2.2_ti2v_v3_cp.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_cp \
    --prompt "A cat walks on the grass, realistic" \
    --negative_prompt "blurred, low quality, static" \
    --output output.mp4
```

### Image-to-Video (I2V)

```bash
python run_wan2.2_ti2v_v3_cp.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_cp \
    --image input.png \
    --prompt "A cat walks on the grass, realistic" \
    --output output_i2v.mp4
```

The I2V pipeline encodes the input image into the first latent frame, then generates the remaining frames via diffusion. The encoder and quant_conv must be compiled (included in `compile_v3_cp.sh` Step 5). If not compiled, the encoder falls back to CPU.

| Argument | Default | Description |
|----------|---------|-------------|
| `--compiled_models_dir` | `/opt/dlami/nvme/compiled_models_v3_cp` | Compiled model directory |
| `--image` | None | Input image for I2V (omit for T2V) |
| `--height` | 512 | Video height |
| `--width` | 512 | Video width |
| `--num_frames` | 81 | Number of frames (81 = 3.4s @ 24fps) |
| `--num_inference_steps` | 50 | Denoising steps (lower = faster but less quality) |
| `--max_sequence_length` | 512 | Max text token length |
| `--output` | `output_v3_cp.mp4` | Output video path |
| `--force_v1_decoder` | false | Force V1 TorchScript decoder |

## Environment

- **Instance**: trn2.48xlarge (8 Neuron cores)
- **Neuron SDK**: PyTorch 2.9 + NxD Inference
- **Virtual env**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- **Storage**: NVMe at `/opt/dlami/nvme` (recommended for compiled models)

### Required Environment Variables (set automatically by run scripts)

```bash
NEURON_RT_NUM_CORES=8
LOCAL_WORLD_SIZE=8
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_LOGICAL_NC_CONFIG=2
```

## Troubleshooting

### "nearest-exact" interpolation error
The custom `autoencoder_kl_wan.py` patches `F.interpolate` to replace `nearest-exact` with `nearest` for Trainium2 compatibility. The compile scripts automatically copy this file to the diffusers package.

### Replica groups assertion error
If you see errors about replica groups `[[0,1,2,3]]` vs expected `[[0,1,2,3],[4,5,6,7]]`, this is the Neuron compiler bug with `DistributedRMSNorm`. The V3 CP transformer uses `local_rms_norm` to avoid this. Ensure you're using the latest `compile_transformer_v3_cp.py`.

### Out of memory
- Compiled models should be stored on NVMe (`/opt/dlami/nvme/`), not the root EBS volume
- The decoder uses bfloat16 to reduce memory. If OOM persists, try `--force_v1_decoder`

### Decoder fallback
The run scripts support automatic fallback: V3 NoCache -> V3 -> V2 -> V1 (TorchScript). If the V3 NoCache decoder wasn't compiled, it will automatically use whatever version is available.
