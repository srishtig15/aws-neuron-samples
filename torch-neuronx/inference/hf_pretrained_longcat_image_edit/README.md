# LongCat-Image-Edit on AWS Trainium2

Run [meituan-longcat/LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) on AWS Trainium2 (trn2.48xlarge) using Neuron SDK.

## Architecture

LongCat-Image-Edit uses a FLUX-like transformer architecture:

| Component | Size | Execution | Notes |
|-----------|------|-----------|-------|
| Transformer | ~6.3B | Neuron (TP=8) | NKI Flash Attention |
| Text Encoder (Qwen2.5-VL) | ~8B | Hybrid | Vision on Neuron, LM on CPU |
| VAE (AutoencoderKL) | ~300M | Neuron | Standard 2D, tiled processing |

**Transformer details:**
- Joint blocks (dual-stream cross-attention) + Single blocks (single-stream)
- 24 attention heads x 128 head dim = 3072 hidden dim
- Pre-computed RoPE (Neuron-compatible, no complex numbers)
- AdaLayerNormZero / AdaLayerNormZeroSingle

## Setup

### 1. NVMe Storage

```bash
sudo bash setup_nvme.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Compile All Components

```bash
bash compile.sh
```

This takes ~1-2 hours and compiles:
- VAE encoder/decoder (512x512 tiles)
- Transformer (TP=8, NKI Flash Attention, 1024x1024)
- Vision encoder (float32, single device)

### 4. Run Inference

```bash
# Image editing
NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
    --image input.png \
    --prompt "change the cat to a dog" \
    --output output.png

# With custom parameters
NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
    --image input.png \
    --prompt "make it a watercolor painting" \
    --output output.png \
    --num_inference_steps 28 \
    --guidance_scale 3.5 \
    --seed 42
```

## Compilation Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Image size | 1024x1024 | Output resolution |
| VAE tile size | 512x512 | For tiled processing |
| TP degree | 8 | 24 heads / 8 = 3 heads/rank |
| Text seq len | 512 | Padded at runtime |
| Img patches | 8192 | 4096 noise + 4096 image (editing) |
| Compiler target | trn2, LNC=2 | TRN2 48xlarge |

## Environment Variables

```bash
NEURON_FUSE_SOFTMAX=1
NEURON_CUSTOM_SILU=1
XLA_DISABLE_FUNCTIONALIZATION=1   # Required for NKI Flash Attention
NEURON_RT_VIRTUAL_CORE_SIZE=2     # TRN2
NEURON_LOGICAL_NC_CONFIG=2        # TRN2
LOCAL_WORLD_SIZE=8                # TP degree
```

## File Structure

```
hf_pretrained_longcat_image_edit/
├── README.md
├── compile.sh                      # Main compilation script
├── setup_nvme.sh                   # NVMe RAID0 setup
├── requirements.txt
├── run_longcat_image_edit.py       # Main inference script
└── neuron_longcat_image_edit/
    ├── __init__.py
    ├── cache_hf_model.py           # Download model
    ├── neuron_parallel_utils.py    # TP sharding utilities
    ├── compile_vae.py              # VAE compilation
    ├── compile_transformer.py      # Transformer compilation (NKI)
    └── compile_text_encoder.py     # Vision encoder compilation
```

## Key Differences from Qwen-Image-Edit

| Aspect | LongCat | Qwen |
|--------|---------|------|
| Transformer | FLUX-like (joint+single blocks) | Uniform 60-block |
| VAE | Standard 2D AutoencoderKL | 3D video-capable |
| RoPE | Real cos/sin (native) | Complex -> real conversion |
| Model size | ~6.3B (transformer) | ~20B |
| Latent channels | 16 (2x2 pack -> 64) | 16 |
