#!/usr/bin/env python3
"""
Debug which Neuron-compiled component is causing issues.

Run with different --use_neuron_xxx flags to isolate the problematic component.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
from PIL import Image

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def load_image(path):
    return Image.open(path).convert('RGB')


def main():
    parser = argparse.ArgumentParser(description="Debug Neuron components")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_debug_component.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)

    # Component selection flags
    parser.add_argument("--use_neuron_transformer", action="store_true",
                        help="Use Neuron-compiled transformer (otherwise CPU)")
    parser.add_argument("--use_neuron_vae", action="store_true",
                        help="Use Neuron-compiled VAE (otherwise CPU)")
    parser.add_argument("--use_neuron_vision", action="store_true",
                        help="Use Neuron-compiled vision encoder (otherwise CPU)")

    args = parser.parse_args()

    print("=" * 60)
    print("Debug Component Test")
    print("=" * 60)
    print(f"  Transformer: {'NEURON' if args.use_neuron_transformer else 'CPU'}")
    print(f"  VAE: {'NEURON' if args.use_neuron_vae else 'CPU'}")
    print(f"  Vision Encoder: {'NEURON' if args.use_neuron_vision else 'CPU'}")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16

    # Override VAE_IMAGE_SIZE
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_pipeline_module
    qwen_pipeline_module.VAE_IMAGE_SIZE = args.height * args.width

    # Load pipeline
    print("\n[1] Loading pipeline...")
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )

    # Configure processor
    target_pixels = args.image_size * args.image_size
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels

    # Setup text encoder with our wrapper
    from neuron_qwen_image_edit.neuron_commons import NeuronTextEncoderWrapper

    cpu_vision_encoder = pipe.text_encoder.model.visual
    cpu_vision_encoder.eval()
    cpu_language_model = pipe.text_encoder.model.language_model
    cpu_language_model.eval().to(dtype)

    # Handle vision encoder
    compiled_vision = None
    use_cpu_vision = not args.use_neuron_vision
    if args.use_neuron_vision:
        vision_path = f"{COMPILED_MODELS_DIR}/vision_encoder/model.pt"
        if os.path.exists(vision_path):
            print(f"  Loading Neuron vision encoder from {vision_path}")
            compiled_vision = torch.jit.load(vision_path)
            use_cpu_vision = False
        else:
            print(f"  WARNING: Vision encoder not found at {vision_path}, using CPU")
            use_cpu_vision = True

    original_text_encoder = pipe.text_encoder
    pipe.text_encoder = NeuronTextEncoderWrapper(
        original_text_encoder=original_text_encoder,
        compiled_vision_encoder=compiled_vision,
        compiled_language_model=None,
        cpu_language_model=cpu_language_model,
        cpu_vision_encoder=cpu_vision_encoder if use_cpu_vision else None,
        image_size=args.image_size,
        max_seq_len=512
    )

    # Handle transformer
    if args.use_neuron_transformer:
        import neuronx_distributed
        from run_qwen_image_edit import NeuronTransformerWrapper

        transformer_path = f"{COMPILED_MODELS_DIR}/transformer"
        if os.path.exists(transformer_path):
            print(f"  Loading Neuron transformer from {transformer_path}")
            compiled_transformer = neuronx_distributed.trace.parallel_model_load(transformer_path)

            patch_h = args.height // 8 // 2
            patch_w = args.width // 8 // 2
            img_shapes = [(2, patch_h, patch_w)]

            original_transformer = pipe.transformer
            pipe.transformer = NeuronTransformerWrapper(
                original_transformer, compiled_transformer, img_shapes,
                expected_num_patches=2 * patch_h * patch_w,
                expected_seq_len=512
            )
        else:
            print(f"  WARNING: Transformer not found at {transformer_path}, using CPU")
    else:
        pipe.transformer = pipe.transformer.to("cpu")

    # Handle VAE
    if args.use_neuron_vae:
        from run_qwen_image_edit import NeuronVAEWrapper
        from neuron_qwen_image_edit.autoencoder_kl_qwenimage_neuron import AutoencoderKLQwenImage as NeuronAutoencoder

        vae_encoder_path = f"{COMPILED_MODELS_DIR}/vae_encoder/model.pt"
        vae_decoder_path = f"{COMPILED_MODELS_DIR}/vae_decoder/model.pt"

        if os.path.exists(vae_encoder_path) and os.path.exists(vae_decoder_path):
            print(f"  Loading Neuron VAE")

            # Create Neuron-compatible VAE
            original_vae_config = pipe.vae.config
            neuron_vae = NeuronAutoencoder(
                base_dim=original_vae_config.base_dim,
                z_dim=original_vae_config.z_dim,
                dim_mult=original_vae_config.dim_mult,
                num_res_blocks=original_vae_config.num_res_blocks,
                attn_scales=original_vae_config.attn_scales,
                temperal_downsample=original_vae_config.temperal_downsample,
                dropout=original_vae_config.dropout,
                input_channels=original_vae_config.input_channels,
                latents_mean=original_vae_config.latents_mean,
                latents_std=original_vae_config.latents_std,
            )
            neuron_vae.load_state_dict(pipe.vae.state_dict())

            compiled_encoder = torch.jit.load(vae_encoder_path)
            compiled_decoder = torch.jit.load(vae_decoder_path)

            pipe.vae = NeuronVAEWrapper(
                neuron_vae, compiled_encoder, compiled_decoder,
                expected_height=args.height, expected_width=args.width
            )
        else:
            print(f"  WARNING: VAE not found, using CPU")
            pipe.vae = pipe.vae.to("cpu")
    else:
        pipe.vae = pipe.vae.to("cpu")

    # Ensure text_encoder is on CPU
    pipe.text_encoder.to("cpu")

    # Monkey-patch _execution_device
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))

    # Load images
    print(f"\n[2] Loading images...")
    source_images = [load_image(p).resize((args.width, args.height)) for p in args.images]
    input_images = source_images[0] if len(source_images) == 1 else source_images

    # Run inference
    print(f"\n[3] Running inference ({args.num_inference_steps} steps)...")
    generator = torch.Generator().manual_seed(args.seed)

    start = time.time()
    result = pipe(
        prompt=args.prompt,
        image=input_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=5.0,
        generator=generator,
    )
    print(f"  Took {time.time() - start:.1f}s")

    # Save
    result.images[0].save(args.output)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
