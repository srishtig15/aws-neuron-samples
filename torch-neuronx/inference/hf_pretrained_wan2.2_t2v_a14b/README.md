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
| Resolution | 480×832, 81 frames |

## Parallelism Strategy

- **TP=4**: Tensor Parallelism splits each transformer across 4 NeuronCores
- **CP=2**: Context Parallelism splits the sequence (32,760 tokens → 16,380/rank)
- **world_size=8**: Uses all 8 NeuronCores on trn2.48xlarge
- **MoE weight swap**: Both transformers share one compiled NEFF; weights are swapped via `NxDModel.replace_weights()` at the timestep boundary

## Pipeline Phases

1. **Text Encoding** (Neuron): Encode prompt with Neuron-compiled UMT5 text encoder (TP=4)
   - Also supports CPU fallback via `--cpu_text_encoder` flag
2. **Denoising** (Neuron): 50 steps with MoE transformer switching
   - Steps 1-16: transformer_1 (high-noise expert)
   - Weight swap via `replace_weights()`
   - Steps 17-50: transformer_2 (low-noise expert)
3. **VAE Decode** (Neuron): Chunked decoder with rolling feat_cache for flicker-free output
   - Auto-detects rolling cache (`decoder_rolling/`) or NoCache (`decoder_nocache/`) mode

## Performance

| Phase | Time |
|-------|------|
| Text Encoding (Neuron) | ~14s (0.2s inference + 13s model load) |
| Text Encoding (CPU) | ~16s |
| Denoising (50 steps + MoE swap) | ~547s |
| VAE Decode (rolling cache) | ~44s (25s decode + 18s model load) |
| **Total** | **~612s** |

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

# Run inference
python run_wan2.2_t2v_a14b.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
    --prompt "A cat walks on the grass, realistic" \
    --output output_t2v.mp4

# With CPU text encoder (slower, use as fallback)
python run_wan2.2_t2v_a14b.py \
    --cpu_text_encoder \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
    --prompt "A cat walks on the grass, realistic" \
    --output output_t2v.mp4
```

## Compilation

`compile.sh` compiles the following components:

| Step | Component | Output |
|------|-----------|--------|
| 1 | Cache HuggingFace model (~126GB) | HF cache dir |
| 2 | Text Encoder (TP=4) | `text_encoder/` |
| 3 | Transformer - high-noise expert (TP=4, CP=2) | `transformer/` |
| 4 | Transformer_2 - low-noise expert (TP=4, CP=2) | `transformer_2/` |
| 5 | VAE Decoder (Rolling Cache) | `decoder_rolling/` |
| 6 | Post-quant conv | `post_quant_conv/` |

The script auto-patches `nearest-exact` → `nearest` in diffusers for Trainium2 compatibility.

## Inference Options

```
--compiled_models_dir     Path to compiled models (default: /opt/dlami/nvme/compiled_models_t2v_a14b)
--height                  Video height (default: 480)
--width                   Video width (default: 832)
--num_frames              Number of frames (default: 81)
--num_inference_steps     Denoising steps (default: 50)
--guidance_scale          CFG guidance scale (default: 5.0)
--prompt                  Text prompt
--negative_prompt         Negative prompt
--output                  Output video path (default: output_t2v_a14b.mp4)
--cpu_text_encoder        Use CPU text encoder instead of Neuron (slower)
```

## VAE Decoder Modes

The inference script auto-detects which decoder mode to use:

| Mode | Directory | Flicker | Transfer/chunk | Notes |
|------|-----------|---------|----------------|-------|
| **Rolling Cache** | `decoder_rolling/` | No | ~3.6GB (1.8GB in + out) | feat_cache carried between chunks as I/O |
| **NoCache** | `decoder_nocache/` | Yes | ~300KB | feat_cache as zero buffers, no temporal context |

Rolling cache is preferred. If `decoder_rolling/nxd_model.pt` exists, it is used automatically.

## GPU Baseline Benchmark

For performance comparison, `run_wan2.2_t2v_a14b_gpu.py` runs the same model on NVIDIA GPUs using diffusers.

### Quick Start (GPU)

```bash
pip install diffusers transformers accelerate

# 480P (official: 326.9s on H100)
python run_wan2.2_t2v_a14b_gpu.py --resolution 480P

# 720P (official: 1041.5s on H100)
python run_wan2.2_t2v_a14b_gpu.py --resolution 720P

# With CPU offload (reduces VRAM, needed for <80GB GPUs)
python run_wan2.2_t2v_a14b_gpu.py --resolution 480P --offload
```

### H100 Benchmark Results

| Resolution | Official (FA3) | Our Script (SDPA) | Our Script (FA4) | Gap |
|------------|---------------|-------------------|-------------------|-----|
| 480P | 326.9s / 41.3GB | 342.8s | TBD | +4.9% |
| 720P | 1041.5s / 59.8GB | 1118.4s | TBD | +7.4% |

Official numbers from [Wan2.2 repo](https://github.com/Wan-Video/Wan2.2) using FlashAttention3 on H100, single GPU with `--offload_model True --convert_model_dtype`.

### Flash-Attn-4 Support (CUDA 13 + Hopper)

On CUDA 13, the standard `flash-attn` v2 package has no pre-built wheels. `flash-attn-4` uses CuTeDSL for runtime JIT kernel compilation, making it compatible with CUDA 13 without building from source.

```bash
# 1. Install flash-attn-4
pip install --pre flash-attn-4

# 2. Patch diffusers to add _flash_4 backend
python patch_diffusers_fa4.py

# 3. Verify
python patch_diffusers_fa4.py --check

# 4. Run with flash-attn-4
DIFFUSERS_ATTN_BACKEND="_flash_4" python run_wan2.2_t2v_a14b_gpu.py --resolution 480P
```

Note: First run with flash-attn-4 triggers JIT kernel compilation (adds a few minutes to warmup). Subsequent runs use cached kernels.

The patch modifies 3 diffusers files (auto-detected path):
- `utils/import_utils.py` — detect `flash_attn.cute` availability
- `utils/__init__.py` — export detection function
- `models/attention_dispatch.py` — register `_flash_4` attention backend

## File Structure

```
hf_pretrained_wan2.2_t2v_a14b/
├── README.md
├── ROLLING_CACHE.md                     # Rolling cache design document
├── compile.sh                           # Master compilation script (Neuron)
├── run_wan2.2_t2v_a14b.py              # Neuron inference script
├── run_wan2.2_t2v_a14b_gpu.py          # GPU inference benchmark
├── patch_diffusers_fa4.py              # Patch diffusers for flash-attn-4
├── requirements.txt
└── neuron_wan2_2_t2v_a14b/
    ├── __init__.py
    ├── cache_hf_model.py               # Download HF model
    ├── compile_transformer.py          # Transformer compilation (TP=4, CP=2)
    ├── compile_text_encoder.py         # UMT5 text encoder compilation
    ├── compile_decoder_nocache.py      # VAE decoder compilation (NoCache)
    ├── compile_decoder_rolling.py      # VAE decoder compilation (Rolling Cache)
    ├── distributed_rmsnorm.py          # Distributed RMSNorm for TP
    ├── neuron_commons.py               # Wrapper classes for Neuron models
    └── neuron_parallel_utils.py        # TP/CP sharding utilities
```
