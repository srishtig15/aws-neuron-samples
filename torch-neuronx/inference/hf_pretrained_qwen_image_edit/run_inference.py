"""
Qwen-Image-Edit-2509 Inference on Trainium2

This script runs the Qwen-Image-Edit-2509 model on Neuron accelerators.
The transformer and VAE are compiled for Neuron, while the text encoder
runs on CPU due to its complexity as a multimodal vision-language model.
"""

import os
import time
import argparse
import torch
import torch_neuronx
import neuronx_distributed
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from neuron_qwen_image_edit.neuron_commons import (
    InferenceTransformerWrapper,
    SimpleWrapper,
)

CACHE_DIR = "qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "compiled_models"


def load_compiled_models(pipe, compiled_models_dir):
    """Load compiled Neuron models and replace pipeline components."""

    # Load compiled transformer
    transformer_path = f"{compiled_models_dir}/transformer"
    if os.path.exists(transformer_path):
        print(f"Loading compiled transformer from {transformer_path}...")
        transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
        transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(
            transformer_path
        )
        pipe.transformer = transformer_wrapper
        print("Transformer loaded successfully!")
    else:
        print(f"Warning: Compiled transformer not found at {transformer_path}")
        print("Using original transformer (will run on CPU)")

    # Load compiled VAE decoder
    decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    if os.path.exists(decoder_path):
        print(f"Loading compiled VAE decoder from {decoder_path}...")
        vae_decoder_wrapper = SimpleWrapper(pipe.vae.decoder)
        vae_decoder_wrapper.model = torch_neuronx.DataParallel(
            torch.jit.load(decoder_path), [0, 1, 2, 3], False
        )
        pipe.vae.decoder = vae_decoder_wrapper
        print("VAE decoder loaded successfully!")
    else:
        print(f"Warning: Compiled VAE decoder not found at {decoder_path}")

    # Load compiled VAE encoder (if exists)
    encoder_path = f"{compiled_models_dir}/vae_encoder/model.pt"
    if os.path.exists(encoder_path):
        print(f"Loading compiled VAE encoder from {encoder_path}...")
        vae_encoder_wrapper = SimpleWrapper(pipe.vae.encoder)
        vae_encoder_wrapper.model = torch_neuronx.DataParallel(
            torch.jit.load(encoder_path), [0, 1, 2, 3], False
        )
        pipe.vae.encoder = vae_encoder_wrapper
        print("VAE encoder loaded successfully!")

    # Load compiled post_quant_conv (if exists)
    post_quant_path = f"{compiled_models_dir}/post_quant_conv/model.pt"
    if os.path.exists(post_quant_path) and hasattr(pipe.vae, 'post_quant_conv'):
        print(f"Loading compiled post_quant_conv from {post_quant_path}...")
        post_quant_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
        post_quant_wrapper.model = torch_neuronx.DataParallel(
            torch.jit.load(post_quant_path), [0, 1, 2, 3], False
        )
        pipe.vae.post_quant_conv = post_quant_wrapper
        print("post_quant_conv loaded successfully!")

    return pipe


def run_inference(args):
    """Run inference with Qwen-Image-Edit-2509."""

    print("=" * 60)
    print("Qwen-Image-Edit-2509 Inference on Trainium2")
    print("=" * 60)
    print()

    # Load pipeline
    print("Loading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=CACHE_DIR
    )
    print("Pipeline loaded!")
    print()

    # Load compiled models
    print("Loading compiled Neuron models...")
    pipe = load_compiled_models(pipe, args.compiled_models_dir)
    print()

    # Load input images if provided
    images = []
    if args.input_images:
        for img_path in args.input_images:
            print(f"Loading input image: {img_path}")
            images.append(Image.open(img_path))

    # Prepare inputs
    inputs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.manual_seed(args.seed),
        "num_images_per_prompt": args.num_images,
    }

    if images:
        inputs["image"] = images

    if args.true_cfg_scale > 0:
        inputs["true_cfg_scale"] = args.true_cfg_scale

    # Warmup run
    print("Running warmup inference...")
    warmup_start = time.time()
    with torch.inference_mode():
        _ = pipe(**inputs)
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")
    print()

    # Actual inference
    print("Running inference...")
    inference_start = time.time()
    with torch.inference_mode():
        output = pipe(**inputs)
    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.2f}s")
    print()

    # Save output images
    for idx, img in enumerate(output.images):
        output_path = f"{args.output_dir}/output_{idx}.png"
        img.save(output_path)
        print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2509 Inference")

    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image editing/generation")
    parser.add_argument("--negative_prompt", type=str, default=" ",
                        help="Negative prompt")
    parser.add_argument("--input_images", type=str, nargs="*",
                        help="Input image paths (1-3 images)")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="True CFG scale")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for generated images")
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    run_inference(args)
