# Wan2.2-T2V-A14B Inference on AWS Trainium2

Run [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) text-to-video inference on AWS Trainium2 (trn2.48xlarge).

## Model Architecture

Wan2.2-T2V-A14B is a **Mixture-of-Experts (MoE)** text-to-video diffusion model with two 14B-parameter WanTransformer3DModel transformers that switch based on denoising timestep:

| Component | Specification |
|-----------|--------------|
| Transformers | 2 x WanTransformer3DModel (14B params each) |
| Hidden dim | 5120 (40 heads x 128 head_dim) |
| Layers | 40 |
| MoE switch | timestep >= 875 -> transformer_1 (high-noise), < 875 -> transformer_2 (low-noise) |
| Text encoder | UMT5-XXL (4096 dim) |
| VAE | AutoencoderKLWan (z_dim=16) |
| Resolution | 480x832 / 720x1280, 81 frames |

## Parallelism Strategy

| | 480P | 720P |
|---|---|---|
| **TP** | 4 | 4 |
| **CP** | 2 | 4 |
| **world_size** | 8 | 16 |
| **Tokens/rank** | 16,380 | 18,900 |
| **NeuronCores (logical)** | 8 (2 chips) | 16 (4 chips) |

## Pipeline Architecture

All Neuron phases run in **isolated subprocesses** for clean HBM lifecycle. This enables reliable multi-inference: each subprocess loads models, runs inference, and exits — OS-level cleanup guarantees full HBM release.

### Phase 1: Text Encoding (Neuron subprocess)
- UMT5-XXL text encoder compiled with TP=4 (world_size=8)
- Subprocess loads NxDModel, encodes prompt + negative prompt, exits
- CPU fallback available via `--cpu_text_encoder` (required for 720P where text encoder world_size conflicts with transformer world_size)

### Phase 2: Denoising (Neuron subprocess, Combined MoE)
- **Single subprocess** handles both MoE transformer phases using `replace_weights()`:
  1. Load transformer_1 NxDModel (68s)
  2. Run 13 high-noise steps with guidance_scale=4.0 (107s)
  3. Swap to transformer_2 weights via `replace_weights()` (61s)
  4. Run 27 low-noise steps with guidance_scale_2=3.0 (221s)
  5. Subprocess exits, HBM fully released
- `replace_weights()` avoids a second `NxDModel.load()` + `to_neuron()`, saving ~10s vs two separate subprocesses
- 720P uses subprocess-based reload instead (NEFF uses 23.94/24GB HBM, no headroom for in-place swap)

### Phase 3: VAE Decode (Neuron subprocess)
- **Stateful rolling decoder**: feat_cache as registered buffers with automatic input-output aliasing
  - 34 cache tensors stay on device (HBM) between chunks — no host-device transfer
  - Only input x (~300KB) transferred per chunk call
  - 2x faster than legacy I/O cache mode (~1.0s/chunk vs ~2.2s/chunk)
- Post-quant conv + chunked rolling decode (2 latent frames per chunk, 11 chunks for 480P)
- 720P: **Parallel tiled Neuron decode** with 8 tiles (2x4 grid of 416x416 patches) on 8 independent NCs
  - Each tile runs in its own subprocess with a world_size=1 stateful decoder
  - All 8 tiles decode simultaneously, ~18x faster than CPU fallback

## Performance

### 480P (480x832, 81 frames, 40 steps)

| Phase | Time | Details |
|-------|------|---------|
| Text Encoding | 22s | 12s model load + 0.4s inference |
| Denoising | 457s | 69s load + 61s swap + 107s phase1 + 221s phase2 |
| VAE Decode | 44s | 24s model load + 11s decode (stateful) + 8s PQC load |
| **Total** | **~544s** | |

Per-step breakdown: ~8.2s/step (1s Neuron forward + ~3s CPU scheduler + ~4s data transfer overhead)

### 720P (720x1280, 81 frames, 40 steps)

| Phase | Time | Details |
|-------|------|---------|
| Text Encoding (CPU) | 5s | |
| Denoising | 1069s | 2x188s loads + ~16.1s/step |
| VAE Decode (parallel tiled) | ~32s | 8 tiles on 8 NCs, ~20s parallel load + ~4s decode + ~1s PQC |
| **Total** | **~1106s** | |

### Comparison with GPU

| | Trn2 480P | H100 FA4 (no offload) | Ratio |
|---|---|---|---|
| Per step | ~8.2s | ~6.85s | 1.2x |
| Total | ~544s | ~278s | 2.0x |

> The gap is primarily from model loading overhead (130s) and CPU scheduler latency between steps.

## Prerequisites

- AWS trn2.48xlarge instance with Neuron SDK
- Python virtualenv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- NVMe storage mounted at `/opt/dlami/nvme` (~200GB required for model cache + compiled models)

## Quick Start

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b
export PYTHONPATH=$(pwd):$PYTHONPATH

# Install dependencies
pip install -r requirements.txt

# Compile all models (~1-2 hours first time)
bash compile.sh

# Run 480P inference
python run_wan2.2_t2v_a14b.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
    --prompt "A cat walks on the grass, realistic" \
    --output output_t2v_480p.mp4

# Run 720P inference (requires separate compilation with --height 720 --width 1280)
python run_wan2.2_t2v_a14b.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b_720p \
    --height 720 --width 1280 --cpu_text_encoder \
    --prompt "A cat walks on the grass, realistic" \
    --output output_t2v_720p.mp4
```

## Compilation

`compile.sh` compiles the following components:

| Step | Component | Output |
|------|-----------|--------|
| 1 | Cache HuggingFace model (~126GB) | HF cache dir |
| 2 | Text Encoder (TP=4) | `text_encoder/` |
| 3 | Transformer - high-noise expert (TP=4, CP=N) | `transformer/` |
| 4 | Transformer_2 - low-noise expert (TP=4, CP=N) | `transformer_2/` |
| 5 | VAE Decoder (Stateful Rolling Cache, 480P) | `decoder_rolling/` |
| 5b | VAE Decoder (world_size=1 tile, for parallel tiled 720P decode) | `decoder_tile_ws1/` |
| 6 | Post-quant conv (480P only) | `post_quant_conv/` |

For 720P, compile with `--height 720 --width 1280` which sets CP=4 (world_size=16). Step 5b compiles a world_size=1 decoder for 416x416 tiles (`decoder_tile_ws1/`). The full-resolution 720P decoder exceeds the instruction limit; parallel tiled decode runs 8 tiles (2x4 grid) on 8 independent NCs with overlap blending. **Note**: Step 5b must NOT use `neuron_parallel_compile` as it overrides world_size.

The script auto-patches `nearest-exact` -> `nearest` in diffusers for Trainium2 compatibility.

## Inference Options

```
--compiled_models_dir     Path to compiled models (default: /opt/dlami/nvme/compiled_models_t2v_a14b)
--height                  Video height (default: 480)
--width                   Video width (default: 832)
--num_frames              Number of frames (default: 81)
--num_inference_steps     Denoising steps (default: 40)
--guidance_scale          High-noise guidance scale (default: 4.0)
--guidance_scale_2        Low-noise guidance scale (default: 3.0)
--prompt                  Text prompt
--negative_prompt         Negative prompt
--output                  Output video path (default: output_t2v_a14b.mp4)
--cpu_text_encoder        Use CPU text encoder instead of Neuron (required for 720P)
--cpu_vae_decoder         Force CPU VAE decoder (auto-detected if no Neuron decoder compiled)
```

## VAE Decoder Modes

The inference script auto-detects which decoder mode to use:

| Mode | Directory | Flicker | Cache Location | Transfer/chunk | Notes |
|------|-----------|---------|----------------|----------------|-------|
| **Stateful Rolling** | `decoder_rolling/` | No | On-device (HBM) | ~300KB (x only) | Default. Cache as registered buffers, auto input-output aliasing |
| **Legacy Rolling** | `decoder_rolling/` | No | Host (CPU) | ~3.6GB (cache I/O) | Fallback if config `"stateful": false` |
| **Parallel Tiled (720P)** | `decoder_tile_ws1/` | No | On-device (HBM) | ~300KB x 8 tiles | 8 parallel ws=1 decoders on 8 NCs, overlap blending |
| **NoCache** | `decoder_nocache/` | Yes | N/A | ~300KB | No temporal context, visible flicker |

Auto-detection priority: full-res rolling/nocache -> parallel tiled (decoder_tile_ws1) -> CPU fallback.

## File Structure

```
hf_pretrained_wan2.2_t2v_a14b/
├── README.md                            # This file (Trainium2 inference)
├── ROLLING_CACHE.md                     # Rolling cache design document
├── GPU_BENCHMARK.md                     # GPU baseline benchmark & flash-attn-4 guide
├── compile.sh                           # Master compilation script (Neuron)
├── run_wan2.2_t2v_a14b.py              # Neuron inference script (subprocess orchestrator)
├── run_wan2.2_t2v_a14b_gpu.py          # GPU inference benchmark
├── patch_diffusers_fa4.py              # Patch diffusers for flash-attn-4
├── requirements.txt
└── neuron_wan2_2_t2v_a14b/
    ├── __init__.py
    ├── cache_hf_model.py               # Download HF model
    ├── compile_transformer.py          # Transformer compilation (TP=4, CP=N)
    ├── compile_text_encoder.py         # UMT5 text encoder compilation
    ├── compile_decoder_nocache.py      # VAE decoder compilation (NoCache)
    ├── compile_decoder_rolling.py      # VAE decoder compilation (Rolling Cache, stateful/legacy)
    ├── text_encoder_worker.py          # Subprocess worker for text encoding
    ├── denoise_worker.py               # Subprocess worker for MoE denoising (single/combined mode)
    ├── decoder_worker.py               # Subprocess worker for VAE decode (stateful/legacy/nocache)
    ├── tiled_decoder_worker.py          # Subprocess worker for single-tile decode (parallel tiled 720P)
    ├── distributed_rmsnorm.py          # Distributed RMSNorm for TP
    ├── neuron_commons.py               # Wrapper classes for Neuron models
    └── neuron_parallel_utils.py        # TP/CP sharding utilities
```
