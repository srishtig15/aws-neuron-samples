# Wan2.2-T2V-A14B Inference on AWS Trainium2

Run [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) text-to-video inference on AWS Trainium2 (trn2.48xlarge).

## Model Architecture

Wan2.2-T2V-A14B is a **Mixture-of-Experts (MoE)** text-to-video diffusion model with two 14B-parameter WanTransformer3DModel transformers that switch based on denoising timestep:

| Component | Specification |
|-----------|--------------|
| Transformers | 2 × WanTransformer3DModel (14B params each) |
| Hidden dim | 5120 (40 heads × 128 head_dim) |
| Layers | 40 |
| MoE switch | timestep ≥ 875 → transformer_1 (high-noise), < 875 → transformer_2 (low-noise) |
| Text encoder | UMT5-XXL (4096 dim) |
| VAE | AutoencoderKLWan (z_dim=16) |
| Resolution | 480×832 / 720×1280, 81 frames |

## Parallelism Strategy

| | 480P | 720P |
|---|---|---|
| **TP** | 4 | 4 |
| **CP** | 2 | 4 |
| **world_size** | 8 | 16 |
| **Tokens/rank** | 16,380 | 18,900 |
| **NeuronCores (logical)** | 8 (2 chips) | 16 (4 chips) |

- **MoE weight swap**:
  - 480P: `replace_weights()` in-process (fast, ~63s)
  - 720P: Subprocess-based reload (NEFF uses 23.94/24GB HBM, no headroom for in-place swap)

## Pipeline Phases

1. **Text Encoding** (Neuron or CPU): Encode prompt with Neuron-compiled UMT5 text encoder (TP=4)
   - Also supports CPU fallback via `--cpu_text_encoder` flag
   - 720P requires `--cpu_text_encoder` (text encoder world_size=8 conflicts with transformer world_size=16)
2. **Denoising** (Neuron): 40 steps with MoE transformer switching
   - Steps 1-13: transformer_1 (high-noise expert, guidance_scale=4.0)
   - Weight swap via `replace_weights()` (480P) or subprocess reload (720P)
   - Steps 14-40: transformer_2 (low-noise expert, guidance_scale_2=3.0)
3. **VAE Decode** (Neuron or CPU): Chunked decoder with rolling feat_cache for flicker-free output
   - 480P: Auto-detects rolling cache (`decoder_rolling/`) or NoCache (`decoder_nocache/`) mode
   - 720P: Tiled Neuron decode with 480P patches (`decoder_rolling_480p/`) or CPU fallback

## Performance

### 480P (480×832, 81 frames)

| Phase | Time |
|-------|------|
| Text Encoding (Neuron) | ~15s (0.2s inference + 12s model load) |
| Denoising (40 steps + MoE swap) | ~466s (~8.2s/step + 62s weight swap) |
| VAE Decode (rolling cache) | ~46s (24s decode + 18s model load) |
| **Inference time** | **~526s** |

### 720P (720×1280, 81 frames)

| Phase | Time |
|-------|------|
| Text Encoding (CPU) | ~5s |
| Denoising (40 steps, subprocess mode) | ~1069s (~16.1s/step + 2×188s model load) |
| VAE Decode (Neuron tiled) | ~145s (106s decode + 36s load + 3s PQC) |
| **Inference time** | **~1320s (~22.0min)** |

> **Note**: Tiled decode uses 4 overlapping 480P patches (2×2 grid) with linear-ramp blending.
> CPU fallback (~585s) is still available via `--cpu_vae_decoder` or if `decoder_rolling_480p/` is not compiled.

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
| 5 | VAE Decoder (Rolling Cache, 480P) | `decoder_rolling/` |
| 5b | VAE Decoder (480P patches, for tiled 720P decode) | `decoder_rolling_480p/` |
| 6 | Post-quant conv (480P only) | `post_quant_conv/` |

For 720P, compile with `--height 720 --width 1280` which sets CP=4 (world_size=16). Step 5b compiles a 480P decoder for tiled 720P decode (`decoder_rolling_480p/`). The full-resolution 720P decoder exceeds the instruction limit; tiled decode uses 480P patches with overlap blending instead.

The script auto-patches `nearest-exact` → `nearest` in diffusers for Trainium2 compatibility.

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

| Mode | Directory | Flicker | Transfer/chunk | Notes |
|------|-----------|---------|----------------|-------|
| **Rolling Cache** | `decoder_rolling/` | No | ~3.6GB (1.8GB in + out) | feat_cache carried between chunks as I/O |
| **Tiled (720P)** | `decoder_rolling_480p/` | No | ~3.6GB × 4 tiles | 480P rolling cache patches with overlap blending |
| **NoCache** | `decoder_nocache/` | Yes | ~300KB | feat_cache as zero buffers, no temporal context |

Rolling cache is preferred. Auto-detection priority: full-res rolling/nocache → tiled 480P → CPU fallback.

## File Structure

```
hf_pretrained_wan2.2_t2v_a14b/
├── README.md                            # This file (Trainium2 inference)
├── ROLLING_CACHE.md                     # Rolling cache design document
├── GPU_BENCHMARK.md                     # GPU baseline benchmark & flash-attn-4 guide
├── compile.sh                           # Master compilation script (Neuron)
├── run_wan2.2_t2v_a14b.py              # Neuron inference script
├── run_wan2.2_t2v_a14b_gpu.py          # GPU inference benchmark
├── patch_diffusers_fa4.py              # Patch diffusers for flash-attn-4
├── requirements.txt
└── neuron_wan2_2_t2v_a14b/
    ├── __init__.py
    ├── cache_hf_model.py               # Download HF model
    ├── compile_transformer.py          # Transformer compilation (TP=4, CP=N)
    ├── compile_text_encoder.py         # UMT5 text encoder compilation
    ├── compile_decoder_nocache.py      # VAE decoder compilation (NoCache)
    ├── compile_decoder_rolling.py      # VAE decoder compilation (Rolling Cache)
    ├── denoise_worker.py              # Subprocess worker for HBM-tight denoising (720P)
    ├── distributed_rmsnorm.py          # Distributed RMSNorm for TP
    ├── neuron_commons.py               # Wrapper classes for Neuron models
    └── neuron_parallel_utils.py        # TP/CP sharding utilities
```
