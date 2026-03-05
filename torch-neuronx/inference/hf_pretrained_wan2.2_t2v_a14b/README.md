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

1. **Text Encoding** (CPU): Encode prompt with UMT5 text encoder
2. **Denoising** (Neuron): 50 steps with MoE transformer switching
   - Steps 1-6: transformer_1 (high-noise expert)
   - Weight swap via `replace_weights()`
   - Steps 7-50: transformer_2 (low-noise expert)
3. **VAE Decode** (CPU): Decode latents with float32 VAE → video frames

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
```

## Compilation

`compile.sh` compiles the following components:

| Step | Component | Output |
|------|-----------|--------|
| 1 | Cache HuggingFace model (~126GB) | HF cache dir |
| 2 | Text Encoder (TP=4) | `text_encoder/` |
| 3 | Transformer - high-noise expert (TP=4, CP=2) | `transformer/` |
| 4 | Transformer_2 - low-noise expert (TP=4, CP=2) | `transformer_2/` |
| 5 | VAE Decoder | `decoder_nocache/` |
| 6 | Post-quant conv | `post_quant_conv/` |

The script auto-patches `nearest-exact` → `nearest` in diffusers for Trainium2 compatibility.

## Inference Options

```
--compiled_models_dir   Path to compiled models (default: /opt/dlami/nvme/compiled_models_t2v_a14b)
--height                Video height (default: 480)
--width                 Video width (default: 832)
--num_frames            Number of frames (default: 81)
--num_inference_steps   Denoising steps (default: 50)
--guidance_scale        CFG guidance scale (default: 5.0)
--prompt                Text prompt
--negative_prompt       Negative prompt
--output                Output video path (default: output_t2v_a14b.mp4)
```

## File Structure

```
hf_pretrained_wan2.2_t2v_a14b/
├── README.md
├── compile.sh                           # Master compilation script
├── run_wan2.2_t2v_a14b.py              # Main inference script
├── requirements.txt
└── neuron_wan2_2_t2v_a14b/
    ├── __init__.py
    ├── cache_hf_model.py               # Download HF model
    ├── compile_transformer.py          # Transformer compilation (TP=4, CP=2)
    ├── compile_text_encoder.py         # UMT5 text encoder compilation
    ├── compile_decoder_nocache.py      # VAE decoder compilation
    ├── distributed_rmsnorm.py          # Distributed RMSNorm for TP
    ├── neuron_commons.py               # Wrapper classes for Neuron models
    └── neuron_parallel_utils.py        # TP/CP sharding utilities
```

## Known Limitations

- Neuron VAE decode (~10s) has flickering artifacts due to NoCache mode (feat_cache is zero buffers, no temporal context between chunks). CPU VAE decode is flicker-free but ~100s. Rolling feat_cache is WIP.
- Total inference time ~560s (denoising ~545s + Neuron VAE ~10s) at 480P 81 frames.
