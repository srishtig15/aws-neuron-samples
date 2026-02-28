# RollingForcing on AWS Trainium2

Adaptation of [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) for AWS Trainium2 (trn2.48xlarge) inference.

RollingForcing uses a DMD-distilled CausalWanModel (Wan2.1-T2V-1.3B) that generates long videos via a rolling window strategy with only 5 denoising steps per block.

**Target**: trn2.48xlarge, venv `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`

## Architecture Overview

### Model: CausalWanModel (1.3B)

| Parameter | Value |
|-----------|-------|
| dim | 1536 |
| ffn_dim | 8960 |
| num_heads | 12 |
| head_dim | 128 |
| num_layers | 30 |
| text_dim | 4096 (UMT5-XXL) |
| text_len | 512 |
| patch_size | (1, 2, 2) |
| Resolution | 480x832 |
| Latent spatial | 60x104 -> patched 30x52 = 1560 tokens/frame |
| max_frames | 21 (latent) -> 81 (pixel via VAE temporal upsampling) |
| Denoising steps | 5 (DMD distilled) |

### Design: Full-Sequence Processing (No KV Cache)

The original RollingForcing uses per-layer KV cache with dynamic rolling/eviction and an attention sink mechanism. This is incompatible with Neuron's static tensor shape requirement. Instead, we process the **full visible sequence** (up to 21 frames = 32760 tokens) in each forward call:

| | Original (GPU) | Ours (Neuron) |
|--|----------------|---------------|
| Model calls per window | 2 (denoise + cache update) | 1 (denoise only) |
| Total calls for 21 frames | 22 | 11 |
| Per-call tokens (Q) | Variable (up to 23400) | Fixed (32760, padded) |
| KV cache management | Dynamic per-layer rolling | None (host-side latents) |
| Attention mask | Causal via KV cache structure | No explicit mask (NKI flash attention) |

## Project Structure

```
hf_pretrained_rolling_forcing/
├── README.md                           # This file
├── CLAUDE.md                           # Project status and notes
├── compile.sh                          # Master compilation script
├── run_rolling_forcing.py              # Main Neuron inference pipeline
├── run_rolling_forcing_gpu.py          # GPU reference benchmark (H100)
├── diagnose.py                         # Diagnostic tests
├── test_single_step.py                 # Single-step transformer isolation test
├── test_cpu_vs_neuron.py               # CPU vs Neuron output comparison
├── test_5step_cpu.py                   # Full 5-step CPU test
├── test_diffusers_baseline.py          # Diffusers baseline comparison
└── neuron_rolling_forcing/
    ├── __init__.py
    ├── cache_hf_model.py               # Download model weights
    ├── compile_transformer.py          # CausalWanModel compilation (CORE)
    ├── compile_text_encoder.py         # UMT5-XXL text encoder
    ├── compile_decoder_nocache.py      # Wan 3D VAE decoder
    ├── neuron_commons.py               # NKI attention, RoPE, utils
    ├── neuron_parallel_utils.py        # TP sharding functions
    └── neuron_rope.py                  # Real-valued RoPE for Wan
```

## Compilation

### Prerequisites

```bash
# On trn2.48xlarge
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install diffusers transformers omegaconf easydict einops safetensors imageio

# Clone RollingForcing repo (needed for model definitions)
cd /tmp && git clone https://github.com/TencentARC/RollingForcing.git
```

### Step 1: Download Models

```bash
python neuron_rolling_forcing/cache_hf_model.py
```

Downloads:
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (text encoder, VAE, base config)
- RollingForcing DMD checkpoint

### Step 2: Compile All Components

```bash
bash compile.sh
```

Compiles three components:

1. **Text Encoder** (UMT5-XXL, TP=8, ~15 min)
2. **VAE Decoder** (Wan 3D VAE, NoCache mode, TP=8, ~5 min)
3. **Transformer** (CausalWanModel, TP=4, world_size=8, ~30-60 min)

Output: `compiled_models/` directory.

## Inference

```bash
python run_rolling_forcing.py \
    --prompt A cat walking gracefully across a sunlit garden path \
    --num_frames 81 \
    --seed 42 \
    --output_path ~/Downloads/output.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | A cat walking... | Text prompt |
| `--num_frames` | 81 | Output video frames (81 = 21 latent frames) |
| `--seed` | 42 | Random seed |
| `--fps` | 16 | Output video FPS |
| `--guidance_scale` | 3.0 | CFG scale (1.0 = no CFG, >1 = amplified) |
| `--negative_prompt` | (Chinese) | Negative prompt for CFG |
| `--decode_cpu` | False | Use CPU Diffusers VAE instead of compiled Neuron VAE |
| `--save_latents` | False | Save raw latent output for debugging |
| `--denoising_step_list` | [1000,800,600,400,200] | Raw denoising step indices |

### Pipeline Flow

1. **Text Encoding**: Tokenize prompt -> compiled UMT5-XXL -> prompt embeddings [1, 512, 4096]
2. **Timestep Warping**: Raw steps [1000, 800, 600, 400, 200] -> warped [1000.0, 952.4, 882.4, 769.2, 555.6]
3. **Rolling Forcing Loop**: 11 windows, each processing up to 5 blocks (15 frames):
   - Build noisy input from cached blocks + fresh noise for new block
   - Prepend anchor frames (finalized clean blocks) for temporal context
   - Assign per-frame timesteps (clean blocks: t=0, noisy blocks: varying t)
   - Pad to 21 frames with noise at max_t
   - Compute RoPE for frame positions
   - Run compiled transformer -> flow prediction
   - Convert flow to x0: `x0 = xt - sigma * flow_pred`
   - Re-noise x0 predictions with fresh noise at next timestep
4. **VAE Decoding**: Latent [1, 21, 16, 60, 104] -> pixel [1, 81, 3, 480, 832]
5. **Save**: Write MP4 video

## Implementation Details

### Neuron Adaptations

| Original Component | Neuron Replacement | Notes |
|--------------------|--------------------|-------|
| Complex RoPE (`torch.polar`) | Real-valued cos/sin | Pre-computed, passed as input |
| `flash_attn` / `flex_attention` | NKI Flash Attention | kernel + causal variant via `flash_fwd` |
| Dynamic KV cache | Eliminated | Full-sequence processing, host-side cache |
| Block-wise causal mask | See Quality section | Various options tested |
| `WanRMSNorm` (global) | `local_rms_norm` | Per-TP-rank normalization, no all-reduce |
| float64 flow->x0 | float32 | Neuron doesn't support float64 |
| `nearest-exact` upsampling | `nearest` | Neuron compatibility patch |

### Tensor Parallelism (TP=4 for transformer)

| Layer | Type | Per-rank Shape |
|-------|------|----------------|
| `self_attn.q/k/v` | ColumnParallel | [1536 -> 384] (3 heads x 128) |
| `self_attn.o` | RowParallel | [384 -> 1536] |
| `cross_attn.q/k/v` | ColumnParallel | [1536 -> 384] |
| `cross_attn.o` | RowParallel | [384 -> 1536] |
| `ffn.0` (up) | ColumnParallel | [1536 -> 2240] |
| `ffn.2` (down) | RowParallel | [2240 -> 1536] |
| `norm_q/norm_k` | local_rms_norm | [384] (manually sharded) |

### RoPE: 3D Positional Encoding

Wan RoPE splits head_dim=128 into three axes:
- Temporal: 44 dims
- Spatial height: 42 dims
- Spatial width: 42 dims

For each token at position (f, h, w):
```
freqs = concat(freqs_t[f], freqs_h[h], freqs_w[w])
rope_cos = cos(freqs), rope_sin = sin(freqs)
```

Padding frames get sequential positions after real frames (not position 0) to avoid RoPE collision.

### Flow Matching Scheduler

```
sigma_warped = shift * sigma / (1 + (shift - 1) * sigma)    # shift=5.0
timestep = sigma_warped * 1000

x0 = xt - sigma * flow_pred                                  # flow to x0
x_next = (1 - sigma_next) * x0 + sigma_next * fresh_noise    # re-noising
```

## Quality Analysis

### GPU Reference Experiments (H100)

Comprehensive experiments were run on H100 to identify the root cause of the quality gap.
The original pipeline (KV cache) serves as the quality reference (pixel_std = 70.2).

| Method | pixel_std | cos_sim vs original | Notes |
|--------|-----------|---------------------|-------|
| **Original GPU (KV cache)** | **70.2** | **1.0** | Reference |
| Neuron padded, CFG=7 | 56.8 | - | Current best Neuron output |
| GPU bidirectional, no pad | 22.4 | 0.186 | SDPA, no mask |
| GPU block-causal, no pad | 22.8 | 0.177 | Per-block SDPA (correct training mask) |
| GPU token-causal, no pad | ~14 | 0.139 | Triangular causal mask |

### Key Findings

1. **Padding dilution is NOT the main issue**: Removing padding makes quality WORSE (22.4 vs 56.8). The padding with noise at max_t actually helps by providing the model additional context.

2. **Attention mask type does not matter**: Bidirectional (22.4), block-causal (22.8), and token-causal (~14) all produce similar poor quality. This rules out attention masking as the root cause.

3. **KV cache is essential for quality**: The original pipeline's two-pass approach provides:
   - **Isolated clean representations**: Clean frames' K/V are computed in a separate cache-update pass (t≈0), never contaminated by noisy frames
   - **Frozen context**: During the denoise pass, cached clean K/V is read-only; only noisy frames' representations are computed fresh
   - Without this isolation, clean frames are reprocessed alongside noisy frames in every window, degrading their representations

4. **The quality gap is fundamental to single-pass processing**: Any approach that processes clean + noisy frames in a single forward pass will suffer, regardless of attention masking, because the model was designed for KV-cache inference.

### Next Steps: KV Cache Implementation

To match original quality, we need to implement KV-cache inference on Neuron. Options under investigation:

- **External KV cache I/O**: Pass per-layer K/V as additional model inputs/outputs (30 layers x 2 tensors each)
- **Custom NKI kernel**: Add `effective_kv_len` parameter to mask padded KV positions
- **Bucketed compilation**: Compile separate models for each KV length bucket
- **Block-causal NKI kernel**: Modify flash_fwd causal predicate for block-level granularity

NKI flash_fwd kernel capabilities:
- Supports different Q/KV lengths
- `logit_bias`: [1,1,seq_q,seq_k] with batch/head broadcasting, but full Q×K matrix required (~2.4GB for 32760²)
- `use_causal_mask`: tile-level skip for token-level causal
- `sliding_window`: causal sliding window

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| Text encoding | ~10.7s | First call includes warmup |
| Denoising (11 windows) | ~22.2s | ~2.0s/window |
| VAE decoding | ~9.9s | 21 latent -> 81 pixel frames |
| **Total** | **~43s** | End-to-end |

## Code Reuse from Wan2.2 TI2V

| Component | Source | Reuse Level |
|-----------|--------|-------------|
| NKI flash attention | compile_transformer_v3_cp.py | Direct copy |
| local_rms_norm() | compile_transformer_v3_cp.py | Direct copy |
| apply_rotary_emb() | compile_transformer_v3_cp.py | Adapted (no CP split) |
| f32Wrapper | neuron_commons.py | Direct copy |
| UMT5 sharding | neuron_parallel_utils.py | Direct copy |
| VAE NoCache decoder | compile_decoder_v3_nocache.py | Adapted for Wan2.1 |
| ModelBuilder pattern | compile_transformer_v3_cp.py | Adapted |

## References

- [RollingForcing (TencentARC)](https://github.com/TencentARC/RollingForcing)
- [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (HuggingFace)
