# Technical Notes: Qwen-Image-Edit on Neuron

This document summarizes key technical challenges and solutions for running Qwen-Image-Edit on AWS Trainium2.

## Architecture Overview

| Component | Execution | Reason |
|-----------|-----------|--------|
| Transformer (20.4B) | Neuron TP=8 | Large compute, benefits from parallelism |
| VAE (~300M) | Neuron DP=8 | Tiled processing for large images |
| Vision Encoder (~1.4B) | CPU (default) | Neuron compilation causes precision loss |
| Language Model (7B) | CPU | GQA 28Q/4KV incompatible with TP=8 |

## Key Fixes

### 1. Modulation Layer Sharding

**Problem**: Transformer `img_mod` and `txt_mod` layers (6.8B params total) were duplicated on every TP rank.

**Solution**: Shard with `ColumnParallelLinear(gather_output=True)` in `neuron_parallel_utils.py`.

**Impact**: Transformer size reduced from ~17GB to ~5.2GB per shard.

### 2. RoPE Implementation

**Problem**: Original RoPE uses complex numbers (unsupported on Neuron) and the initial Neuron implementation had incorrect cos/sin expansion.

**Solution** (`neuron_rope.py`):
- Use real-number operations instead of complex
- Use `repeat_interleave` for correct interleaved expansion: `[c0, c1, ...] -> [c0, c0, c1, c1, ...]`
- Handle multiple `img_shapes` formats from pipeline

### 3. M-RoPE Position IDs

**Problem**: Qwen2.5-VL requires 3D position_ids `[3, batch, seq]` for Multimodal RoPE. Text and image tokens have different position patterns.

**Solution** (`neuron_commons.py:_get_rope_index`):
- Text tokens: sequential positions (same for t, h, w)
- Image tokens: 3D grid positions based on spatial layout

### 4. Vision Encoder Processor Configuration

**Problem**: Processor dynamically determines output size based on input image, causing mismatch with compiled vision encoder.

**Solution** (`run_qwen_image_edit.py`):
```python
target_pixels = args.image_size * args.image_size
pipe.processor.image_processor.min_pixels = target_pixels
pipe.processor.image_processor.max_pixels = target_pixels
```

### 5. VAE Interpolation Mode

**Problem**: Original VAE uses `F.interpolate(mode='nearest-exact')` which isn't supported on Neuron.

**Solution** (`autoencoder_kl_qwenimage_neuron.py`): Use `mode='nearest'`.

### 6. BFloat16 for Parallel Layers

**Problem**: `neuronx_distributed` parallel layers default to float32, doubling model size.

**Solution**: Always specify `dtype=torch.bfloat16` when creating parallel layers.

## GQA Incompatibility with TP=8

The Qwen2.5-VL language model uses Grouped Query Attention:
- 28 Q heads, 4 KV heads
- Group size = 7 (each KV head serves 7 Q heads)

With TP=8:
- Q heads per rank: 28/8 = 3.5 (not integer, needs padding to 32)
- KV heads per rank: 4/8 = 0.5 (cannot shard evenly)

The Q-KV mapping cannot be preserved with TP=8. Valid TP degrees: 1, 2, or 4.

**Solution**: Run language model on CPU.

## Performance Notes

### Current Performance (Optimized)

Performance on TRN2 for 1024x1024 with 2-image merge (40 steps, CFG enabled):

| Component | Time |
|-----------|------|
| Vision Encoder (CPU, first call) | ~21s |
| Vision Encoder (CPU, cached) | ~0.7s |
| Language Model (CPU) | ~1-4s |
| VAE encode | ~0.4s |
| VAE decode | ~1.6s |
| **Transformer** | **~1.9s/step** |

**Total inference time**: ~190s (40 steps × 2 CFG passes)

### Performance Comparison

| Platform | Transformer Speed | Relative |
|----------|------------------|----------|
| H100 (without Flash Attention) | ~0.75s/step | 1.0x |
| TRN2 (optimized) | ~1.9s/step | 2.5x slower |

### Compiler Optimizations Applied

The following compiler flags provide ~32% speedup over baseline:

```python
compiler_flags = """
    --target=trn2
    --lnc=2
    --model-type=transformer              # Transformer-specific optimizations
    -O1                                   # Optimization level
    --auto-cast=none                      # Preserve bfloat16 precision
    --tensorizer-options='--enable-ccop-compute-overlap'  # Key: overlap comm/compute
    --enable-fast-loading-neuron-binaries
"""
```

**Key optimization**: `--enable-ccop-compute-overlap` allows tensor parallel communication (all-reduce) to overlap with computation, significantly reducing synchronization overhead.

### NKI Flash Attention Status

Attempted to use NKI Flash Attention kernel (`neuronxcc.nki._private_kernels.attention.attention_isa_kernel`), which is used by Flux on Neuron.

**Result**: NKI kernel is not compatible with `parallel_model_trace` (V1 API) due to XLA tracing limitations - the kernel's output parameter cannot be modified during tracing.

**Workaround**: Would require migration to `ModelBuilder` API (V2), but V2 has RoPE buffer constant-folding issues that produce incorrect outputs.

**Current Status**: Using standard SDPA attention with compiler optimizations.

## File Reference

| File | Purpose |
|------|---------|
| `neuron_commons.py` | Text encoder wrapper, M-RoPE position IDs |
| `neuron_parallel_utils.py` | Tensor parallelism utilities, modulation sharding |
| `neuron_rope.py` | Neuron-compatible RoPE implementation |
| `autoencoder_kl_qwenimage_neuron.py` | VAE with nearest interpolation |
| `compile_transformer.py` | Transformer compilation with TP=8 |
| `compile_text_encoder.py` | Vision encoder compilation |
| `compile_vae.py` | VAE encoder/decoder compilation |
