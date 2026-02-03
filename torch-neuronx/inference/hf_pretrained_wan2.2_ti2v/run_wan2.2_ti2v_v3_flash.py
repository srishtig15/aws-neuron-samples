"""
Wan2.2 TI2V Inference with NKI Flash Attention (V3 Flash).

This script uses:
- NxDModel.load() for text_encoder (V2 API)
- NxDModel.load() for transformer with NKI Flash Attention (V3 Flash API)
- NxDModel.load() for decoder and post_quant_conv (V2 API) if available
- Falls back to torch.jit.load() for decoder and post_quant_conv (V1 API)

Key features:
- NKI Flash Attention for optimal performance
- TP=8, world_size=8
- No Context Parallel (simpler deployment)

Usage:
    NEURON_RT_NUM_CORES=8 python run_wan2.2_ti2v_v3_flash.py --compiled_models_dir compiled_models_v3_flash
"""
# IMPORTANT: Set environment variables BEFORE any imports
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["WORLD_SIZE"] = "8"
os.environ["RANK"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import argparse
import json
import numpy as np
import random
import time
import torch
import torch_neuronx

from neuronx_distributed import NxDModel, NxDParallelState
from safetensors.torch import load_file

from neuron_wan2_2_ti2v.neuron_commons_v2 import InferenceTextEncoderWrapperV2
from neuron_wan2_2_ti2v.neuron_commons import SimpleWrapper, DecoderWrapper, DecoderWrapperV2, PostQuantConvWrapperV2


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_model_config(model_path):
    """Load model configuration from config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    """Load TP sharded weights from safetensors files."""
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)
        sharded_weights.append(ckpt)
    return sharded_weights


def load_duplicated_weights(model_path, world_size):
    """
    Load single checkpoint and duplicate for all ranks.

    For models like decoder that don't use actual TP sharding,
    we load tp0 checkpoint and duplicate for all world_size ranks.

    Args:
        model_path: Path to the compiled model directory
        world_size: Number of ranks to duplicate to

    Returns:
        List of world_size checkpoint dicts (all identical)
    """
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)

    # Duplicate for all ranks
    sharded_weights = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded_weights.append(ckpt)

    return sharded_weights


class InferenceTransformerWrapperV3Flash(torch.nn.Module):
    """
    Wrapper for transformer with NKI Flash Attention (V3 Flash).
    """

    def __init__(self, transformer, nxd_model, rotary_emb_cos, rotary_emb_sin):
        super().__init__()
        self.transformer = transformer  # Original transformer for config access
        self.nxd_model = nxd_model
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        self.cache_context = transformer.cache_context

        # Pre-computed RoPE
        self.rotary_emb_cos = rotary_emb_cos
        self.rotary_emb_sin = rotary_emb_sin

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, return_dict=False, **kwargs):
        """Forward with pre-computed RoPE."""
        # Call NxDModel with RoPE
        if hasattr(self.nxd_model, 'inference'):
            output = self.nxd_model.inference(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb_cos=self.rotary_emb_cos,
                rotary_emb_sin=self.rotary_emb_sin,
            )
        else:
            output = self.nxd_model(
                hidden_states,
                timestep,
                encoder_hidden_states,
                self.rotary_emb_cos,
                self.rotary_emb_sin,
            )

        # Handle tuple return
        if isinstance(output, (tuple, list)):
            output = output[0]

        return output


def load_transformer_v3_flash(compiled_models_dir, pipe):
    """
    Load V3 Flash compiled transformer.

    Args:
        compiled_models_dir: Directory containing compiled models
        pipe: Original pipeline for config access

    Returns:
        InferenceTransformerWrapperV3Flash instance
    """
    v3_flash_path = f"{compiled_models_dir}/transformer_v3_flash"

    # Load config
    config = load_model_config(v3_flash_path)
    tp_degree = config["tp_degree"]

    print(f"Loading V3 Flash transformer (TP={tp_degree})...")

    # Load checkpoints
    checkpoints = load_sharded_weights(v3_flash_path, tp_degree)

    # Load NxDModel
    nxd_model_path = os.path.join(v3_flash_path, "nxd_model.pt")
    nxd_model = NxDModel.load(nxd_model_path)
    nxd_model.set_weights(checkpoints)
    nxd_model.to_neuron()

    # Load pre-computed RoPE
    rope_cache_path = os.path.join(v3_flash_path, "rope_cache.pt")
    rope_cache = torch.load(rope_cache_path)
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)
    print(f"  Loaded RoPE: cos={rotary_emb_cos.shape}, sin={rotary_emb_sin.shape}")

    # Create wrapper
    wrapper = InferenceTransformerWrapperV3Flash(
        transformer=pipe.transformer,
        nxd_model=nxd_model,
        rotary_emb_cos=rotary_emb_cos,
        rotary_emb_sin=rotary_emb_sin,
    )

    print("V3 Flash transformer loaded.")
    return wrapper


# Defaults
DEFAULT_COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models_v3_flash"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
SEED = 42


def main(args):
    set_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Load base pipeline
    print("Loading base pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    compiled_models_dir = args.compiled_models_dir
    seqlen = args.max_sequence_length

    # Load Text Encoder (V2)
    print("\nLoading text encoder (V2)...")
    text_encoder_dir = f"{compiled_models_dir}/text_encoder_v2"
    text_encoder_wrapper = InferenceTextEncoderWrapperV2(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    text_encoder_config = load_model_config(text_encoder_dir)
    text_encoder_nxd = NxDModel.load(os.path.join(text_encoder_dir, "nxd_model.pt"))
    text_encoder_weights = load_sharded_weights(text_encoder_dir, text_encoder_config["tp_degree"])
    text_encoder_nxd.set_weights(text_encoder_weights)
    text_encoder_nxd.to_neuron()
    text_encoder_wrapper.t = text_encoder_nxd
    print("Text encoder loaded.")

    # Load Transformer (V3 Flash)
    print("\nLoading transformer (V3 Flash with NKI)...")
    transformer_wrapper = load_transformer_v3_flash(compiled_models_dir, pipe)

    # Load Decoder - check for V2 first, fall back to V1
    decoder_v2_path = f"{compiled_models_dir}/decoder_v2"
    decoder_v1_path = f"{compiled_models_dir}/decoder/model.pt"

    if os.path.exists(decoder_v2_path):
        print("\nLoading decoder (V2)...")
        vae_decoder_wrapper = DecoderWrapperV2(pipe.vae.decoder)
        decoder_nxd = NxDModel.load(os.path.join(decoder_v2_path, "nxd_model.pt"))
        decoder_config = load_model_config(decoder_v2_path)
        decoder_world_size = decoder_config.get("world_size", 8)

        # Decoder uses single checkpoint duplicated for all ranks (no actual TP sharding)
        decoder_weights = load_duplicated_weights(decoder_v2_path, decoder_world_size)
        decoder_nxd.set_weights(decoder_weights)
        decoder_nxd.to_neuron()

        vae_decoder_wrapper.nxd_model = decoder_nxd
        print("Decoder (V2) loaded.")
    else:
        print("\nLoading decoder (V1)...")
        vae_decoder_wrapper = DecoderWrapper(pipe.vae.decoder)
        vae_decoder_wrapper.model = torch.jit.load(decoder_v1_path)
        print("Decoder (V1) loaded.")

    # Load post_quant_conv - check for V2 first, fall back to V1
    pqc_v2_path = f"{compiled_models_dir}/post_quant_conv_v2"
    pqc_v1_path = f"{compiled_models_dir}/post_quant_conv/model.pt"

    if os.path.exists(pqc_v2_path):
        print("\nLoading post_quant_conv (V2)...")
        vae_post_quant_conv_wrapper = PostQuantConvWrapperV2(pipe.vae.post_quant_conv)
        pqc_nxd = NxDModel.load(os.path.join(pqc_v2_path, "nxd_model.pt"))
        pqc_config = load_model_config(pqc_v2_path)
        pqc_world_size = pqc_config.get("world_size", 8)

        # post_quant_conv uses single checkpoint duplicated for all ranks
        pqc_weights = load_duplicated_weights(pqc_v2_path, pqc_world_size)
        pqc_nxd.set_weights(pqc_weights)
        pqc_nxd.to_neuron()

        vae_post_quant_conv_wrapper.nxd_model = pqc_nxd
        print("post_quant_conv (V2) loaded.")
    else:
        print("\nLoading post_quant_conv (V1)...")
        vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
        vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
            torch.jit.load(pqc_v1_path), [0, 1, 2, 3], False
        )
        print("post_quant_conv (V1) loaded.")

    # Replace pipeline components
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    pipe.vae.decoder = vae_decoder_wrapper
    pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper

    prompt = args.prompt
    negative_prompt = args.negative_prompt

    # Warmup
    print("\nStarting warmup inference...")
    start = time.time()
    output_warmup = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=seqlen,
        generator=torch.Generator().manual_seed(SEED + 1000)
    ).frames[0]
    end = time.time()
    print(f"Warmup time: {end - start:.2f}s")

    # Reset generator
    generator = torch.Generator().manual_seed(SEED)

    # Main inference
    print("\nStarting main inference...")
    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=seqlen,
        generator=generator
    ).frames[0]
    end = time.time()

    inference_time = end - start
    per_step_time = inference_time / args.num_inference_steps
    print(f"\nInference time: {inference_time:.2f}s")
    print(f"Per step: {per_step_time:.3f}s")
    print(f"Output frames: {len(output)}")

    # Save video
    output_path = args.output
    export_to_video(output, output_path, fps=24)
    print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V Inference with NKI Flash Attention")
    parser.add_argument("--compiled_models_dir", type=str, default=DEFAULT_COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max text sequence length")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic",
                        help="Text prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="Negative prompt")
    parser.add_argument("--output", type=str, default="output_v3_flash.mp4", help="Output video path")
    args = parser.parse_args()

    # Initialize parallel state with world_size=8 and TP=8 (no Context Parallel)
    # This must wrap the entire execution to ensure process groups are available
    print("Initializing NxDParallelState (world_size=8, TP=8)...")
    with NxDParallelState(world_size=8, tensor_model_parallel_size=8):
        main(args)
