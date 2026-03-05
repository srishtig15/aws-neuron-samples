# LongCat-Image-Edit on AWS Trainium2

Adapt [meituan-longcat/LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) for AWS Neuron inference on **trn2.48xlarge**.

Based on the [Qwen-Image-Edit Neuron adaptation](https://github.com/whn09/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_qwen_image_edit) reference implementation.

## Performance

| Machine | Config | Total Time | Per Step | Quality |
|---------|--------|------------|----------|---------|
| **Trn2** (trn2.48xlarge) | All Neuron, **CFG Parallel** | **20.39s** | 0.41s | Good |
| **Trn2** (trn2.48xlarge) | All Neuron, Context Parallel | 22.39s | 0.45s | Good |
| **H100** (single GPU, bf16) | Full GPU | 23.61s | 0.47s | Reference |

Test: 1024x1024 output, guidance_scale=4.5, 50 steps.

### Quality Validation

| Component Config | enc_std | Time | Quality |
|-----------------|---------|------|---------|
| All Neuron (VE + LM + Transformer + VAE 1024) | 4.63 | 20.63s | Good |
| CPU VE + Neuron LM + Transformer + VAE (1024) | 4.17 | 22.53s | Good |
| H100 GPU (default) | 4.41 | 23.61s | Reference |

## Architecture

LongCat-Image-Edit is a FLUX-style image editing model:

| Component | Model | Neuron Parallelism |
|-----------|-------|-------------------|
| Vision Encoder | Qwen2.5-VL ViT (32 blocks) | TP=4, float32 |
| Language Model | Qwen2.5-VL LM (28 layers) | TP=4, world_size=8 |
| Transformer (CP) | LongCatImageTransformer2DModel (10 dual + 20 single stream) | TP=4, CP=2, world_size=8 |
| Transformer (CFG) | LongCatImageTransformer2DModel (10 dual + 20 single stream) | TP=4, DP=2, world_size=8, batch=2 |
| VAE | 2D AutoencoderKL | Single device (1024x1024, no tiling) |

> **Note**: All components run on Neuron by default. The Vision Encoder uses native SDPA
> (matching the [Qwen-Image-Edit reference](https://github.com/whn09/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_qwen_image_edit)).
> Use `--cpu_vision_encoder` to fall back to CPU VE if needed.

## CFG Parallel vs Context Parallel

Both modes use TP=4, world_size=8 on the same hardware. The second parallelism dimension (DP=2) is used differently:

| Aspect | Context Parallel (CP) | CFG Parallel |
|--------|----------------------|--------------|
| Scatter dimension | dim=1 (sequence) | dim=0 (batch) |
| Calls per step | 2 (neg + pos sequential) | 1 (neg + pos batched) |
| K/V All-Gather | Yes (every attention layer) | No |
| Compile batch_size | 1 | 2 |
| RoPE scatter | Yes (split positions) | No (same positions) |
| Best for | guidance_scale = 1 (no CFG) | guidance_scale > 1 (~9% faster) |

CFG Parallel batches negative + positive prompt into a single `batch_size=2` transformer call. Each DP rank processes one batch item with full sequence length, eliminating the K/V all-gather overhead at all 30 attention layers (10 dual + 20 single stream blocks).

## Key Implementation Details

### M-RoPE Position IDs (Critical)

The Qwen2.5-VL Language Model uses 3D Multimodal Rotary Position Embeddings (M-RoPE) with temporal, height, and width dimensions. The `NeuronTextEncoderWrapper` must use the **original model's `get_rope_index()` method** to compute correct position IDs. A custom reimplementation will produce wrong results (positions up to ~601 vs incorrect range of ~93), causing severe quality degradation (cartoon-style output instead of photorealistic).

### VL Processor Resolution

When using CPU Vision Encoder, the image processor uses HuggingFace default resolution (matching H100 behavior). When using compiled Neuron VE, the resolution is fixed to the compiled dimension (e.g., 448x448).

### Vision Encoder Compilation

The Vision Encoder uses native `F.scaled_dot_product_attention` (no monkey-patching), matching the verified Qwen-Image-Edit reference. Earlier attempts with BMM-based SDPA replacement caused severe accuracy degradation (cosine=0.12 vs CPU).

### VAE Compilation

The VAE decoder is compiled for full 1024x1024 output (decoder input `[1, 16, 128, 128]`), eliminating the need for tiled decoding and avoiding tile seam artifacts.

### Transformer Text Sequence Length

The transformer is compiled with `text_seq_len=1024` to handle the full text encoding (typically 770-838 tokens with image tokens). Previous compilation with `text_seq_len=512` caused text truncation and quality loss.

## Prerequisites

- **Instance**: trn2.48xlarge (64 NeuronCores, 1.5TB device memory)
- **Virtual env**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
  - PyTorch 2.9, neuronx-cc 2.22, neuronx-distributed 0.16
- **NVMe**: Mount RAID at `/opt/dlami/nvme/` (run `setup_nvme.sh`)

## Quick Start

### 1. Setup

```bash
# Mount NVMe RAID
sudo bash setup_nvme.sh

# Activate virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

```bash
python neuron_longcat_image_edit/cache_hf_model.py
```

### 3. Compile All Components

```bash
# Compile with Context Parallel (default)
./compile.sh

# Compile with CFG Parallel (recommended when guidance_scale > 1, ~9% faster)
./compile.sh cfg

# Custom dimensions:
# ./compile.sh [cp|cfg] <height> <width> <image_size> <max_seq_len>
# ./compile.sh cfg 1024 1024 448 1024
```

Compilation takes ~60-90 minutes total. Compiled models are saved to `/opt/dlami/nvme/compiled_models/`.

### 4. Run Inference

```bash
# CFG Parallel (recommended, fastest)
NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
    --image assets/test.png \
    --prompt "将猫变成狗" \
    --seed 43 \
    --use_cfg_parallel \
    --output output.png

# Context Parallel (default)
NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
    --image assets/test.png \
    --prompt "将猫变成狗" \
    --seed 43 \
    --output output.png

# With warmup (for benchmarking):
NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
    --image assets/test.png \
    --prompt "将猫变成狗" \
    --seed 43 \
    --use_cfg_parallel \
    --warmup \
    --output output.png
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Input image path |
| `--prompt` | (required) | Edit instruction |
| `--output` | `output_edited.png` | Output image path |
| `--height` | 1024 | Output height |
| `--width` | 1024 | Output width |
| `--num_inference_steps` | 50 | Denoising steps |
| `--guidance_scale` | 4.5 | Guidance scale |
| `--seed` | 42 | Random seed |
| `--negative_prompt` | `" "` | Negative prompt |
| `--image_size` | 448 | Vision encoder image size |
| `--use_cfg_parallel` | false | Use CFG Parallel transformer (batches neg+pos, ~9% faster) |
| `--cpu_vision_encoder` | false | Use CPU vision encoder for better accuracy |
| `--warmup` | false | Run warmup inference first |
| `--compiled_models_dir` | `/opt/dlami/nvme/compiled_models` | Path to compiled models |

## File Structure

```
├── run_longcat_image_edit.py               # Main Neuron inference script
├── run_longcat_image_edit_gpu.py           # GPU reference script
├── compile.sh                              # Master compilation script
├── setup_nvme.sh                           # NVMe RAID setup
├── requirements.txt
├── longcat_h100_ref_test.py                # H100 reference test
├── longcat_h100_matched.py                 # H100 test with matched 448x448 VL settings
├── longcat_benchmark_h100.py               # H100 benchmark
└── neuron_longcat_image_edit/
    ├── cache_hf_model.py                   # Download model + install diffusers
    ├── neuron_commons.py                   # NeuronTextEncoderWrapper, NKI attention
    ├── neuron_parallel_utils.py            # FLUX-specific TP sharding
    ├── neuron_rope.py                      # 3-axis RoPE pre-computation
    ├── compile_transformer.py              # FLUX transformer (TP=4, CP=2)
    ├── compile_transformer_cfg.py          # FLUX transformer (TP=4, DP=2, CFG Parallel)
    ├── compile_vae.py                      # 2D AutoencoderKL (1024x1024)
    ├── compile_vision_encoder.py           # Qwen2.5-VL ViT (TP=4)
    └── compile_language_model.py           # Qwen2.5-VL LM (TP=4)
```

## Debugging History & Lessons Learned

1. **M-RoPE position IDs**: Custom `_get_rope_index` produced completely wrong position IDs (range [0,93] vs correct [0,601]). Always use `original_text_encoder.model.get_rope_index()`.
2. **VL processor resolution**: Must match between Trn2 and H100 for comparable results. CPU VE mode should use default resolution.
3. **Text sequence truncation**: `text_seq_len=512` is insufficient for LongCat (770-838 tokens typical). Use 1024.
4. **VAE tiling seams**: Avoid tiling by compiling VAE decoder for full output size (1024x1024).
5. **Vision Encoder SDPA**: BMM-based SDPA monkey-patch caused cosine=0.12 divergence. Using native SDPA (matching Qwen reference) works correctly.

## H100 Reference

To run the same test on H100 for comparison:

```bash
# On H100 machine
/opt/pytorch/bin/python3 longcat_h100_ref_test.py
```

## Known Issues

- **Sequence length**: Language Model compiled for max_seq_len=1024. Prompts producing longer sequences will be truncated.
- **Neuron LM precision**: bf16 precision over 28 layers causes minor numerical drift (cosine=0.897 vs CPU). This is acceptable for image generation quality.
