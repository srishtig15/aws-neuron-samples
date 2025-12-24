"""
Wan2.2 TI2V Inference using Model Builder V2 API.

This script uses NxDModel.load() instead of the deprecated parallel_model_load.

Usage:
    python run_wan2.2_ti2v_latency_optimized_v2.py --compiled_models_dir compiled_models
"""
# imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import argparse
import numpy as npy
import os
import random
import time
import torch

from neuronx_distributed import NxDModel

from neuron_wan2_2_ti2v.neuron_commons_v2 import (
    InferenceTextEncoderWrapperV2,
    InferenceTransformerWrapperV2,
    SimpleWrapperV2,
    DecoderWrapperV2,
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

    # V2 model paths
    compiled_models_dir = args.compiled_models_dir
    text_encoder_model_path = f"{compiled_models_dir}/text_encoder_v2/nxd_model.pt"
    transformer_model_path = f"{compiled_models_dir}/transformer_v2/nxd_model.pt"
    decoder_model_path = f"{compiled_models_dir}/decoder_v2/nxd_model.pt"
    post_quant_conv_model_path = f"{compiled_models_dir}/post_quant_conv_v2/nxd_model.pt"

    seqlen = args.max_sequence_length

    # Load Text Encoder using NxDModel
    print("Loading text encoder (V2)...")
    text_encoder_wrapper = InferenceTextEncoderWrapperV2(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    text_encoder_nxd = NxDModel.load(text_encoder_model_path)
    text_encoder_nxd.to_neuron()
    text_encoder_wrapper.t = text_encoder_nxd
    print("Text encoder loaded.")

    # Load Transformer using NxDModel
    print("Loading transformer (V2)...")
    transformer_wrapper = InferenceTransformerWrapperV2(pipe.transformer)
    transformer_nxd = NxDModel.load(transformer_model_path)
    transformer_nxd.to_neuron()
    transformer_wrapper.transformer = transformer_nxd
    print("Transformer loaded.")

    # Load Decoder using NxDModel
    print("Loading decoder (V2)...")
    vae_decoder_wrapper = DecoderWrapperV2(pipe.vae.decoder)
    decoder_nxd = NxDModel.load(decoder_model_path)
    decoder_nxd.to_neuron()
    vae_decoder_wrapper.model = decoder_nxd
    print("Decoder loaded.")

    # Load post_quant_conv using NxDModel
    print("Loading post_quant_conv (V2)...")
    vae_post_quant_conv_wrapper = SimpleWrapperV2(pipe.vae.post_quant_conv)
    post_quant_conv_nxd = NxDModel.load(post_quant_conv_model_path)
    post_quant_conv_nxd.to_neuron()
    vae_post_quant_conv_wrapper.model = post_quant_conv_nxd
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
