"""
Wan2.2 TI2V Inference using Model Builder V2 API (Hybrid).

This script uses:
- NxDModel.load() for text_encoder and transformer (V2 API)
- torch.jit.load() for decoder and post_quant_conv (V1 API, due to list input limitation)

Usage:
    python run_wan2.2_ti2v_latency_optimized_v2.py --compiled_models_dir compiled_models
"""
# imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import argparse
import json
import numpy as npy
import os
import random
import time
import torch
import torch_neuronx

from neuronx_distributed import NxDModel
from safetensors.torch import load_file

from neuron_wan2_2_ti2v.neuron_commons_v2 import (
    InferenceTextEncoderWrapperV2,
    InferenceTransformerWrapperV2,
)
# Import V1 wrappers for decoder (due to list input limitation in V2 API)
from neuron_wan2_2_ti2v.neuron_commons import (
    SimpleWrapper,
    DecoderWrapper,
)


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    npy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_model_config(model_path):
    """Load model configuration from config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    """Load sharded weights from safetensors files."""
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)
        sharded_weights.append(ckpt)
    return sharded_weights


# Defaults - can be overridden via command line
DEFAULT_COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"  # V2 scripts default to this
HUGGINGFACE_CACHE_DIR = "wan2.2_ti2v_hf_cache_dir"
SEED = 42


def main(args):
    set_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Load base pipeline
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    # Model paths (V2 for text_encoder/transformer, V1 for decoder/post_quant_conv)
    compiled_models_dir = args.compiled_models_dir
    text_encoder_dir = f"{compiled_models_dir}/text_encoder_v2"
    transformer_dir = f"{compiled_models_dir}/transformer_v2"
    # Note: decoder and post_quant_conv use V1 paths (model.pt) due to list input limitation in V2 API

    seqlen = args.max_sequence_length

    # Load Text Encoder using NxDModel
    print("Loading text encoder (V2)...")
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

    # Load Transformer using NxDModel
    print("Loading transformer (V2)...")
    transformer_wrapper = InferenceTransformerWrapperV2(pipe.transformer)
    transformer_config = load_model_config(transformer_dir)
    transformer_nxd = NxDModel.load(os.path.join(transformer_dir, "nxd_model.pt"))
    transformer_weights = load_sharded_weights(transformer_dir, transformer_config["tp_degree"])
    transformer_nxd.set_weights(transformer_weights)
    transformer_nxd.to_neuron()
    transformer_wrapper.transformer = transformer_nxd
    print("Transformer loaded.")

    # Load Decoder using V1 API (torch.jit.load)
    # Note: Decoder uses feat_cache (list input) which V2 API doesn't support
    print("Loading decoder (V1 - torch.jit)...")
    decoder_model_path = f"{compiled_models_dir}/decoder/model.pt"
    vae_decoder_wrapper = DecoderWrapper(pipe.vae.decoder)
    vae_decoder_wrapper.model = torch.jit.load(decoder_model_path)
    print("Decoder loaded.")

    # Load post_quant_conv using V1 API (torch.jit.load with DataParallel)
    print("Loading post_quant_conv (V1 - torch.jit)...")
    post_quant_conv_model_path = f"{compiled_models_dir}/post_quant_conv/model.pt"
    vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
    vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
        torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False  # Use for trn2
    )
    print("post_quant_conv loaded.")

    # Replace pipeline components with compiled versions
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

    # Reset generator for main inference
    generator = torch.Generator().manual_seed(SEED)

    # Main inference
    print("\nStarting main inference (with fixed seed)...")
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
    print(f"Inference time: {end - start:.2f}s")
    print(f"Output shape: {output.shape}")
    print(f"Output frames: {len(output)}")

    # Save video
    output_path = args.output
    export_to_video(output, output_path, fps=24)
    print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V Inference using Model Builder V2")
    parser.add_argument("--compiled_models_dir", type=str, default=DEFAULT_COMPILED_MODELS_DIR,
                        help="Directory containing compiled V2 models")
    parser.add_argument("--height", type=int, default=512, help="Video height (must match compiled model)")
    parser.add_argument("--width", type=int, default=512, help="Video width (must match compiled model)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames (must match compiled model)")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max sequence length for text encoder")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic",
                        help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="Negative prompt")
    parser.add_argument("--output", type=str, default="output_v2.mp4", help="Output video path")
    args = parser.parse_args()

    main(args)
