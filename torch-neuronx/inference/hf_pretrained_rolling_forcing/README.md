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

### Key Design Decision: Full-Sequence Processing (No KV Cache)

The original RollingForcing uses per-layer KV cache with dynamic rolling/eviction and an attention sink mechanism. This is incompatible with Neuron's static tensor shape requirement. Instead, we process the **full visible sequence** (up to 21 frames = 32760 tokens) in each forward call:

| | Original (GPU) | Ours (Neuron) |
|--|----------------|---------------|
| Model calls per window | 2 (denoise + cache update) | 1 (denoise only) |
| Total calls for 21 frames | 22 | 11 |
| Per-call tokens (Q) | Variable (up to 23400) | Fixed (32760, padded) |
| KV cache management | Dynamic per-layer rolling | None (host-side latents) |
| Attention mask | Causal via KV cache | No explicit mask (NKI flash attention) |

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

This compiles three components:

1. **Text Encoder** (UMT5-XXL, TP=8, ~15 min)
   - Same architecture as Wan2.2, adapted from existing compilation code
   - Pre-computes attention bias per TP rank
   - Wraps LayerNorm in f32Wrapper for precision

2. **VAE Decoder** (Wan 3D VAE, NoCache mode, TP=8, ~5 min)
   - Processes 2 latent frames at a time (CACHE_T=2 requirement)
   - 32 feat_cache buffers registered as zero buffers
   - Frame-by-frame decoding with temporal upsampling (4x)

3. **Transformer** (CausalWanModel, TP=4, world_size=8, ~30-60 min)
   - NKI Flash Attention (kernel: `AttentionMMSoftmaxMMWithoutSwap`)
   - Real-valued RoPE (cos/sin) replacing complex-valued `torch.polar`
   - Per-frame timestep modulation preserved
   - Sequence padded to power of 2 (32760 -> 32768) for NKI

Output: `compiled_models/` directory with all compiled artifacts.

## Inference

```bash
python run_rolling_forcing.py \
    --prompt "A cat walking gracefully across a sunlit garden path" \
    --num_frames 81 \
    --seed 42 \
    --output_path ~/Downloads/output.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | "A cat walking..." | Text prompt |
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
   - Assign per-frame timesteps (clean blocks: low t, noisy blocks: high t)
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
| `flash_attn` / `flex_attention` | NKI Flash Attention | `AttentionMMSoftmaxMMWithoutSwap` kernel |
| Dynamic KV cache | Eliminated | Full-sequence processing, host-side cache |
| Block-wise causal mask | No explicit mask | NKI kernel processes full sequence |
| `WanRMSNorm` (global) | `local_rms_norm` | Per-TP-rank normalization, no all-reduce |
| float64 flow->x0 | float32 | Neuron doesn't support float64 |

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

Wan's RoPE splits head_dim=128 into three axes:
- Temporal: 44 dims
- Spatial height: 42 dims
- Spatial width: 42 dims

For each token at position (f, h, w):
```
freqs = concat(freqs_t[f], freqs_h[h], freqs_w[w])
rope_cos = cos(freqs), rope_sin = sin(freqs)
```

The interleaved format uses `repeat_interleave(2)` to match Wan's `view_as_complex` pattern.

### Flow Matching Scheduler

```
sigma_warped = shift * sigma / (1 + (shift - 1) * sigma)    # shift=5.0
timestep = sigma_warped * 1000

# Flow to x0 conversion:
x0 = xt - sigma * flow_pred

# Re-noising for next step:
x_next = (1 - sigma_next) * x0 + sigma_next * fresh_noise
```

## Debugging & Known Issues

### Current Status

The pipeline runs end-to-end with correct shapes and reasonable performance (~22s for denoising), but the output video quality has issues.

### Issue: Grid Artifacts at 2x2 Patch Level

**Symptom**: Output video shows a visible 2x2 grid pattern overlaid on the content.

**Root Cause Analysis**:

1. **Not a Neuron compilation issue**: CPU model produces the same pattern (verified in `test_cpu_vs_neuron.py`, cosine sim 0.995 between CPU and Neuron output)

2. **Not a causal mask issue**: Tested both causal and non-causal attention on CPU - same pattern in both (cosine sim 0.995)

3. **Not a VAE issue**: Decoding pure noise through the VAE produces clean output (no grid). The grid only appears with model-generated latents.

4. **Not in latent patch boundaries**: Analysis showed no systematic difference at 2x2 patch boundaries in latent space (across/within ratios all ~1.0)

5. **Within-patch sub-pixel bias**: The 4 sub-pixels within each 2x2 patch have systematic position-dependent offsets (up to +/-0.49). This is caused by `head_linear.bias` dominating when the model produces near-constant hidden states:
   ```
   Model output:  d(0,0)=-0.11, d(0,1)=-0.37, d(1,0)=+0.33, d(1,1)=+0.15
   Random noise:  d(0,0)=-0.02, d(0,1)=+0.01, d(1,0)-0.01, d(1,1)=+0.02
   ```

6. **Key insight**: The within-patch bias is a SYMPTOM, not the cause. A properly working model produces spatial content that overwhelms the bias. The model's predictions are too weak/uniform, allowing the constant bias term to become visible.

### Hypotheses for Weak Predictions

1. **Missing KV cache context**: The original uses a second model call per window to update KV cache with clean (timestep=0) representations. Our full-sequence approach feeds re-noised blocks instead. The model was trained expecting clean KV cache context.

2. **Missing anchor frames**: After block 0 exits the rolling window (window 5+), subsequent windows lack the first block's clean representation. The original maintains this as an "attention sink" in the KV cache.

3. **No causal mask**: The original model uses causal attention (each frame only attends to past frames). Our NKI attention has no mask. While CPU testing showed similar results, the model may rely on causal structure during multi-step denoising.

4. **Precision differences**: The original uses float64 for flow->x0 conversion. Our bf16/float32 pipeline may accumulate errors over 5 denoising steps.

### Diagnostic Scripts

- `diagnose.py --test unpatchify`: Verify unpatchify round-trip correctness
- `diagnose.py --test cpu_forward`: Single forward pass on CPU with original model
- `test_single_step.py`: Run compiled transformer at single timestep, decode with CPU VAE
- `test_cpu_vs_neuron.py`: Compare CPU manual forward vs compiled Neuron output

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| Text encoding | ~10.7s | First call, includes warmup |
| Denoising (11 windows) | ~22.2s | ~2.0s/window |
| VAE decoding | ~9.9s | 21 latent -> 81 pixel frames |
| **Total** | **~43s** | End-to-end |

## Code Reuse from Wan2.2 TI2V

| Component | Source | Reuse Level |
|-----------|--------|-------------|
| NKI flash attention | `compile_transformer_v3_cp.py` | Direct copy |
| `local_rms_norm()` | `compile_transformer_v3_cp.py` | Direct copy |
| `apply_rotary_emb()` | `compile_transformer_v3_cp.py` | Adapted (no CP split) |
| `f32Wrapper` | `neuron_commons.py` | Direct copy |
| UMT5 sharding | `neuron_parallel_utils.py` | Direct copy |
| VAE NoCache decoder | `compile_decoder_v3_nocache.py` | Adapted for Wan2.1 |
| ModelBuilder pattern | `compile_transformer_v3_cp.py` | Adapted |

## References

- [RollingForcing Paper](https://arxiv.org/abs/xxxx) (TencentARC)
- [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (HuggingFace)
- Wan2.2 TI2V Neuron adaptation: `/Users/henanwan/Documents/workspace/wan2.2-ti2v-neuron/code/`
