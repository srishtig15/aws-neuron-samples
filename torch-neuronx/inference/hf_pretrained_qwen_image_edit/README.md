# Qwen-Image-Edit on AWS Trainium2

[中文文档](README_CN.md) | English

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

Five compilation APIs are available:

| API | Script | Speed | Notes |
|-----|--------|-------|-------|
| **V3 CP (Fastest)** | `compile_transformer_v3_cp.py` | **~0.77s/step** | Context Parallel (TP=4, CP=2) + NKI Flash Attention |
| V1 Flash | `compile_transformer_v1_flash.py` | ~1.2s/step | parallel_model_trace + NKI Flash Attention |
| V2 Flash | `compile_transformer_v2_flash.py` | ~1.2s/step | ModelBuilder + NKI Flash Attention |
| V2 | `compile_transformer_v2.py` | ~1.2s/step | ModelBuilder API |
| V1 | `compile_transformer.py` | ~2.4s/step | parallel_model_trace API |

**Note**: V3 CP achieves H100-comparable performance (~0.77s/step vs H100's ~0.75s/step) by using Context Parallel to split sequence across 2 data parallel ranks, allowing each rank to process half the sequence while all-gathering K/V for full attention context.

```bash
# Compile V3 CP (fastest, Context Parallel + NKI Flash Attention)
./compile.sh v3_cp

# Or compile V1 Flash (uses NKI Flash Attention)
./compile.sh v1_flash

# Or compile V2 Flash (ModelBuilder + NKI)
./compile.sh v2_flash

# Or compile V2 (ModelBuilder)
./compile.sh v2

# Or compile V1 only
./compile.sh v1
```

Default compilation settings:
- Output image size: 1024x1024
- VAE tile size: 512x512 (fixed, uses tiled processing)
- Max sequence length: 1024
- TP degree: 8
- Patch multiplier: 3 (for 2-image merging)

### 3. Run Inference

```bash
# Two-image merging with V3 CP (fastest, Context Parallel)
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo" \
    --use_v3_cp

# Two-image merging with V1 Flash (NKI Flash Attention)
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo" \
    --use_v1_flash

# Two-image merging with V2 (ModelBuilder)
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo" \
    --use_v2

# Two-image merging with V1
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people into a wedding photo"

# Single image editing (requires patch_multiplier=2)
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --patch_multiplier 2 \
    --use_v1_flash

# With custom CFG scale
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "change the background to a beach" \
    --true_cfg_scale 6.0 \
    --use_v3_cp
```

**Note**: CFG (Classifier-Free Guidance) runs the transformer twice sequentially per step, not with batch_size=2.

## Project Structure

```
hf_pretrained_qwen_image_edit/
├── README.md                      # English documentation
├── README_CN.md                   # Chinese documentation
├── setup_nvme.sh                  # NVMe RAID0 mount script
├── compile.sh                     # Main compilation script
├── run_qwen_image_edit.py         # Inference script
├── neuron_qwen_image_edit/
│   ├── cache_hf_model.py          # Download model from HuggingFace
│   ├── compile_vae.py             # VAE compilation
│   ├── compile_transformer.py     # Transformer V1 compilation (parallel_model_trace)
│   ├── compile_transformer_v2.py  # Transformer V2 compilation (ModelBuilder)
│   ├── compile_transformer_v1_flash.py  # Transformer V1 Flash (NKI Flash Attention)
│   ├── compile_transformer_v2_flash.py  # Transformer V2 Flash (ModelBuilder + NKI)
│   ├── compile_transformer_v3_cp.py     # Transformer V3 CP (Context Parallel + NKI)
│   ├── compile_language_model_v3.py     # Language Model V3 (TP=4, for V3 CP)
│   ├── compile_text_encoder.py    # Text encoder compilation
│   ├── neuron_commons.py          # Common utilities and wrappers
│   ├── neuron_parallel_utils.py   # Tensor parallelism utilities
│   ├── neuron_rope.py             # Neuron-compatible RoPE implementation
│   └── autoencoder_kl_qwenimage_neuron.py  # Neuron-compatible VAE
├── tests/                         # Unit tests
└── compiled_models/               # Output directory for compiled models
    ├── vae_encoder/
    ├── vae_decoder/
    ├── transformer/               # V1: TP=8 sharded transformer
    ├── transformer_v2/            # V2: ModelBuilder compiled transformer
    ├── transformer_v1_flash/      # V1 Flash: NKI Flash Attention
    ├── transformer_v2_flash/      # V2 Flash: ModelBuilder + NKI
    ├── transformer_v3_cp/         # V3 CP: Context Parallel (TP=4, CP=2) + NKI
    ├── language_model_v3/         # Language Model V3: TP=4 (for V3 CP)
    └── vision_encoder/            # Single device
```

## Technical Details

### Model Architecture and Execution

| Component | Total Params | Execution | Notes |
|-----------|-------------|-----------|-------|
| Transformer (V3 CP) | 20.43B | **TP=4, CP=2** on Neuron | ~10.4 GB/shard, H100-comparable speed |
| Transformer (V1/V2) | 20.43B | TP=8 on Neuron | ~5.2 GB/shard |
| Language Model (V3) | 7.07B | **TP=4** on Neuron | ~4.1 GB/shard, used with V3 CP transformer |
| Language Model (V1/V2) | 7.07B | CPU | GQA 28Q/4KV incompatible with TP=8 |
| Vision Encoder | ~1.4B | Compiled for Neuron (single device), runs on CPU by default | CPU default for higher accuracy |
| VAE | ~300M | DP=8 on Neuron | Tiled processing for large images |

**Language Model on Neuron (V3)**: With V3 CP, the language model can now run on Neuron using TP=4. This is a perfect fit for GQA: 28 Q heads / 4 = 7 heads per rank, 4 KV heads / 4 = 1 head per rank. Use `--use_v3_language_model` with `--use_v3_cp`.

**Why Language Model runs on CPU (V1/V2)**: The Qwen2.5-VL language model uses Grouped Query Attention with 28 Q heads and 4 KV heads (group size = 7). With TP=8, the Q-KV mapping cannot be preserved correctly. Valid TP degrees are only 1, 2, or 4.

**Why Vision Encoder defaults to CPU**: The Vision Encoder IS compiled for Neuron, but runs on CPU by default (`--cpu_vision_encoder`) because the compiled version can have precision loss that gets amplified by the language model, potentially causing lower quality outputs. Use `--neuron_vision_encoder` to run on Neuron for faster speed at the cost of some accuracy.

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

#### 5. V2 Pre-computed RoPE

V2 uses ModelBuilder API which requires RoPE frequencies to be passed as input tensors (not computed inside the model). This avoids XLA constant-folding issues:

```python
# V1: RoPE computed inside model (causes XLA issues with ModelBuilder)
# V2/V1 Flash: RoPE pre-computed from original model and passed as input
vid_freqs, txt_freqs = pipe.transformer.pos_embed(video_fhw, max_txt_seq_len=text_seq_len)
img_rotary_emb = torch.stack([vid_freqs.real, vid_freqs.imag], dim=-1)  # [num_patches, 64, 2]
txt_rotary_emb = torch.stack([txt_freqs.real, txt_freqs.imag], dim=-1)  # [text_seq, 64, 2]
```

The RoPE is cached during compilation and loaded at inference time.

#### 6. V3 CP with Context Parallel

V3 CP achieves H100-comparable performance by combining Context Parallel with Tensor Parallel:

**Architecture**: TP=4, CP=2 (world_size=8)
- **Tensor Parallel (TP=4)**: Shards model weights across 4 devices
- **Context Parallel (CP=2)**: Each CP rank processes half the sequence

**Key implementation details**:

1. **SPMDRank for Runtime Rank Detection**: Uses `neuronx_distributed.parallel_layers.layers.SPMDRank` to get the correct global rank at runtime (not trace time). This is critical for proper scatter/gather operations.

```python
from neuronx_distributed.parallel_layers.layers import SPMDRank

class NeuronQwenTransformerV3CP(nn.Module):
    def __init__(self, ...):
        self.global_rank = SPMDRank(world_size=world_size)
        self.data_parallel_group = parallel_state.get_data_parallel_group()

    def forward(self, hidden_states, ...):
        # Compute DP rank at runtime
        dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.tp_degree)
        # Scatter input based on DP rank
        hidden_states = scatter_to_process_group_spmd(hidden_states, dim=1, rank=dp_rank, ...)
```

2. **K/V All-Gather**: Each CP rank gathers full K/V from all CP ranks to see the complete context, while only computing attention for its local query portion.

```python
# In attention module
if self.context_parallel_enabled:
    # Gather full K/V across CP group
    key = gather_from_tensor_model_parallel_region_with_dim(
        key, dim=2, process_group=self.data_parallel_group
    )
    value = gather_from_tensor_model_parallel_region_with_dim(
        value, dim=2, process_group=self.data_parallel_group
    )
```

3. **Output All-Gather**: After attention, the outputs are all-gathered to reconstruct the full sequence.

**Performance gain**: ~1.56x speedup over V1 Flash (0.77s/step vs 1.2s/step) because:
- Each rank processes only 50% of the query sequence
- Communication overhead is amortized by NKI Flash Attention efficiency

#### 7. Language Model V3 on Neuron

With V3 CP transformer using TP=4, the language model can also use TP=4 to run on Neuron instead of CPU:

**Why TP=4 is perfect for Language Model**:
- Q heads: 28 / 4 = 7 heads per rank (evenly divisible)
- KV heads: 4 / 4 = 1 head per rank (evenly divisible)
- No padding or replication needed!

**Compilation**: The language model is compiled using ModelBuilder API with the same `world_size=8` as the V3 CP transformer:

```bash
# Compile V3 CP (includes language model V3)
./compile.sh v3_cp
```

**Usage**:
```bash
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "combine these two people" \
    --use_v3_cp \
    --use_v3_language_model
```

**Note**: Language Model V3 requires V3 CP transformer (both use world_size=8). When using V1/V2/V1 Flash/V2 Flash (TP=8), the language model must run on CPU.

#### 8. V1 Flash with NKI Flash Attention

V1 Flash combines the best of both approaches:
- Uses `parallel_model_trace` API (like V1) which supports NKI kernels
- Uses pre-computed RoPE (like V2) to avoid XLA issues
- Uses NKI Flash Attention kernel for hardware-optimized attention

**Key implementation details**:

1. **Disable XLA Functionalization**: NKI kernels require in-place output modification, which conflicts with XLA's functionalization. Setting `XLA_DISABLE_FUNCTIONALIZATION=1` is critical.

2. **Custom Attention Module**: Replaces diffusers' `Attention` class with `NKIQwenAttention` that directly calls NKI kernel:

```python
# NKI Flash Attention wrapper (same as Flux implementation)
def nki_flash_attention(query, key, value):
    # Input: [B, H, S, D]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16)
    _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    return attn_output.reshape((bs, n_head, q_len, d_head))
```

3. **NKI Kernel Import**: Uses the same import path as Flux for compatibility:

```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
_flash_fwd_call = nki_jit()(attention_isa_kernel)
```

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
| `--use_v3_cp` | False | Use V3 CP transformer (Context Parallel + NKI, fastest) |
| `--use_v3_language_model` | False | Use V3 language model on Neuron (requires V3 CP) |
| `--use_v1_flash` | False | Use V1 Flash transformer (NKI Flash Attention) |
| `--use_v2_flash` | False | Use V2 Flash transformer (ModelBuilder + NKI Flash Attention) |
| `--use_v2` | False | Use V2 transformer (ModelBuilder) |
| `--height` | 1024 | Output image height (must match compiled model) |
| `--width` | 1024 | Output image width (must match compiled model) |
| `--patch_multiplier` | 2 | 2=single image editing, 3=two-image merge |
| `--image_size` | 448 | Vision encoder input size |
| `--max_sequence_length` | 1024 | Text sequence length (must match compilation) |
| `--num_inference_steps` | 40 | Denoising steps |
| `--true_cfg_scale` | 4.0 | CFG scale (runs transformer twice per step) |
| `--cpu_vision_encoder` | True | Run vision encoder on CPU (default, higher accuracy) |
| `--neuron_vision_encoder` | False | Run vision encoder on Neuron (faster, may have precision loss) |
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

### Checkpoint Size Too Large (master_weight issue)

After compilation, if checkpoint files are ~2x larger than expected (e.g., transformer 80GB instead of 40GB, language model 28GB instead of 14GB), the `shard_checkpoint()` function saved both `master_weight` (full unsharded) and `weight` (sharded) tensors.

**Solution**: Remove `master_weight` tensors from checkpoints:

```python
from safetensors.torch import load_file, save_file
import os

def cleanup_checkpoint(weights_dir):
    """Remove master_weight tensors from sharded checkpoints."""
    for rank in range(4):
        path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors")
        data = dict(load_file(path))
        # Remove master_weight tensors
        cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}
        save_file(cleaned, path)
        print(f"tp{rank}: {len(data)} -> {len(cleaned)} tensors")

# Clean up transformer V3 CP
cleanup_checkpoint("/path/to/compiled_models/transformer_v3_cp/weights")

# Clean up language model V3
cleanup_checkpoint("/path/to/compiled_models/language_model_v3/weights")
```

This reduces checkpoint size by ~50% without affecting functionality.

### Missing inv_freq Tensors (Language Model V3)

```
RuntimeError: Missing weight tensor with key language_model.language_model.rotary_emb.inv_freq
```

**Cause**: The `inv_freq` buffers for rotary embeddings are not included in `state_dict()` by default.

**Solution**: Add `inv_freq` buffers to checkpoints from the original model:

```python
from safetensors.torch import load_file, save_file
from diffusers import QwenImageEditPlusPipeline
import torch

# Load original model
pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
lm = pipe.text_encoder.model.language_model

# Collect inv_freq buffers
inv_freq_buffers = {}
for name, buf in lm.named_buffers():
    if 'inv_freq' in name:
        full_key = f"language_model.language_model.{name}"
        inv_freq_buffers[full_key] = buf.to(torch.bfloat16).clone()

# Add to all TP checkpoints
for rank in range(4):
    path = f"weights/tp{rank}_sharded_checkpoint.safetensors"
    data = dict(load_file(path))
    data.update(inv_freq_buffers)
    save_file(data, path)
```

## Performance

### Benchmark Results (1024x1024, 2-image merge, 40 steps with CFG)

| Platform | Transformer Speed | Total Time |
|----------|------------------|------------|
| H100 (without Flash Attention) | ~0.75s/step | ~60s |
| **TRN2 V3 CP (Context Parallel + NKI)** | **~0.77s/step** | **~62s** |
| TRN2 V1 Flash (NKI) | ~1.2s/step | ~96s |
| TRN2 V2 Flash (ModelBuilder + NKI) | ~1.2s/step | ~96s |
| TRN2 V2 (ModelBuilder) | ~1.2s/step | ~96s |
| TRN2 V1 (parallel_model_trace) | ~2.4s/step | ~190s |

**V3 CP achieves H100-comparable performance!** By using Context Parallel (CP=2) with Tensor Parallel (TP=4), each rank only processes half the sequence, achieving ~1.56x speedup over V1 Flash.

### Performance Analysis

#### Why V1 → V2 achieved 2x speedup?

The key difference is **RoPE (Rotary Position Embedding) handling**:

| Version | RoPE Handling | Overhead |
|---------|---------------|----------|
| V1 | Computed inside model every forward pass | High - complex number operations, position encoding |
| V2/V1 Flash/V2 Flash | Pre-computed once, passed as input | Low - just tensor passing |

**Pre-computed RoPE is the main performance improvement** (2x speedup), not the compilation API or attention implementation.

#### Why V2 ≈ V1 Flash ≈ V2 Flash?

| Version | RoPE | Attention | Speed |
|---------|------|-----------|-------|
| V1 | Inside model | Standard SDPA | ~2.4s/step |
| V2 | **Pre-computed** | Standard SDPA | ~1.2s/step |
| V1 Flash | **Pre-computed** | NKI Flash | ~1.2s/step |
| V2 Flash | **Pre-computed** | NKI Flash | ~1.2s/step |

**Key Findings**:
1. **Pre-computed RoPE** is the dominant optimization (V1 → V2: 2x speedup)
2. **NKI Flash Attention vs Compiler-optimized SDPA**: No significant difference - Neuron compiler already optimizes SDPA very well
3. **Compilation API (parallel_model_trace vs ModelBuilder)**: Minimal impact on final performance

### V1 vs V2 vs V1 Flash vs V2 Flash vs V3 CP Comparison

| Aspect | V1 | V2 | V1 Flash | V2 Flash | **V3 CP (Fastest)** |
|--------|-----|-----|----------|----------|---------------------|
| Compilation API | `parallel_model_trace` | `ModelBuilder` | `parallel_model_trace` | `ModelBuilder` | `ModelBuilder` |
| Attention | Standard SDPA | Standard SDPA | NKI Flash | NKI Flash | **NKI Flash** |
| RoPE Handling | Inside model | Pre-computed | Pre-computed | Pre-computed | Pre-computed |
| Parallelism | TP=8 | TP=8 | TP=8 | TP=8 | **TP=4, CP=2** |
| Speed | ~2.4s/step | ~1.2s/step | ~1.2s/step | ~1.2s/step | **~0.77s/step** |
| Key Advantage | Simple | Pre-computed RoPE | NKI | Both | **Context Parallel** |

**Recommendation**: Use **V3 CP** for fastest performance (H100-comparable). Use V1 Flash if you need simpler debugging or don't want Context Parallel complexity.

### Compiler Optimizations

The following compiler flags are used for optimal performance:

```python
--model-type=transformer      # Transformer-specific optimizations
-O1                           # Optimization level
--auto-cast=none              # Preserve bfloat16 precision
--tensorizer-options='--enable-ccop-compute-overlap'  # Overlap communication with computation
```

The `--enable-ccop-compute-overlap` flag is particularly important for tensor parallelism, as it allows all-reduce operations to overlap with computation, reducing synchronization overhead.

### Component Timing Breakdown

#### V3 CP with V3 Language Model (Fastest)

| Component | Time |
|-----------|------|
| Vision Encoder (CPU, first call) | ~21s |
| Vision Encoder (CPU, cached) | ~0.7s |
| **Language Model (Neuron V3)** | **~0.5s** |
| VAE Encode | ~0.4s |
| **Transformer V3 CP (40 steps × 2 CFG)** | **~62s** |
| VAE Decode | ~1.6s |

#### V1 Flash / V2 with CPU Language Model

| Component | Time |
|-----------|------|
| Vision Encoder (CPU, first call) | ~21s |
| Vision Encoder (CPU, cached) | ~0.7s |
| Language Model (CPU) | ~1-4s |
| VAE Encode | ~0.4s |
| **Transformer (40 steps × 2 CFG)** | **~96s** |
| VAE Decode | ~1.6s |

The transformer is the main bottleneck, accounting for ~80% of total inference time. V3 Language Model on Neuron is ~2-8x faster than CPU.

## Known Limitations

1. **Fixed dimensions**: Models are compiled for specific dimensions. Different sizes require recompilation.
2. **Language model (V1/V2)**: Runs on CPU due to GQA architecture (28Q/4KV heads incompatible with TP=8). Use V3 CP with V3 Language Model to run on Neuron.
3. **Sequence length**: Must match between compilation and inference.
4. **NKI Flash Attention**: Requires `XLA_DISABLE_FUNCTIONALIZATION=1` to work with both parallel_model_trace (V1 Flash) and ModelBuilder (V2 Flash). Without this flag, NKI kernels fail with "immutable output parameter" error.
5. **Language Model V3**: Only compatible with V3 CP transformer (both use world_size=8, TP=4).

## References

- [Qwen-Image-Edit Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## License

This code is provided for research and educational purposes. Please refer to the original model license for usage terms.
