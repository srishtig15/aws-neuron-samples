"""
Qwen-Image-Edit-2509 Inference Script for AWS Trainium2

This script runs the Qwen-Image-Edit model ENTIRELY on Neuron devices.
All components (Text Encoder, Transformer, VAE) run on Trainium2.

Components:
- Text Encoder (Qwen2.5-VL): Vision encoder + Language model
- Transformer: QwenImageTransformer2DModel (TP=8)
- VAE: Encoder and Decoder

Usage:
    # Single image editing:
    python run_qwen_image_edit.py --images input.jpg --prompt "change the sky to sunset"

    # Multi-image editing (1-3 images):
    python run_qwen_image_edit.py --images img1.jpg img2.jpg --prompt "combine these images"
"""

import os
import argparse
import random
import time

import numpy as np
import torch
import torch_neuronx
import neuronx_distributed
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

# Import Neuron-compatible VAE
from neuron_qwen_image_edit.autoencoder_kl_qwenimage_neuron import (
    AutoencoderKLQwenImage as NeuronAutoencoder
)
from neuron_qwen_image_edit.neuron_commons import NeuronTextEncoderWrapper


# Constants
COMPILED_MODELS_DIR = "compiled_models"
HUGGINGFACE_CACHE_DIR = "qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
SEED = 42


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set to: {seed}")


class NeuronTransformerWrapper(torch.nn.Module):
    """
    Wrapper for compiled transformer model on Trainium2.
    """
    def __init__(self, original_transformer, compiled_transformer, img_shapes):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.compiled_transformer = compiled_transformer
        self.img_shapes = img_shapes

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, img_shapes=None, return_dict=False, **kwargs):
        """
        Forward pass using compiled transformer on Neuron.
        """
        # Run on compiled Neuron model
        output = self.compiled_transformer(
            hidden_states,
            encoder_hidden_states,
            timestep
        )

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output[0])
        return output


class NeuronVAEWrapper(torch.nn.Module):
    """
    Wrapper for VAE with compiled encoder and decoder on Trainium2.
    """
    def __init__(self, original_vae, compiled_encoder, compiled_decoder,
                 compiled_quant_conv=None, compiled_post_quant_conv=None,
                 expected_height=512, expected_width=512):
        super().__init__()
        self.config = original_vae.config
        self.dtype = original_vae.dtype

        # Compiled models - ALL run on Neuron
        self.compiled_encoder = compiled_encoder
        self.compiled_decoder = compiled_decoder
        self.compiled_quant_conv = compiled_quant_conv
        self.compiled_post_quant_conv = compiled_post_quant_conv

        # Scaling factors - convert to tensors for broadcasting
        # Shape: (1, z_dim, 1, 1) for proper broadcasting with latents
        if isinstance(original_vae.latents_mean, list):
            self.latents_mean = torch.tensor(original_vae.latents_mean).view(1, -1, 1, 1)
        else:
            self.latents_mean = original_vae.latents_mean
        if isinstance(original_vae.latents_std, list):
            self.latents_std = torch.tensor(original_vae.latents_std).view(1, -1, 1, 1)
        else:
            self.latents_std = original_vae.latents_std

        # z_dim for shape calculations
        self.z_dim = original_vae.config.z_dim

        # Expected input size for compiled model
        self.expected_height = expected_height
        self.expected_width = expected_width

    def encode(self, x, return_dict=True):
        """Encode images to latents on Neuron."""
        # Check and resize input if needed
        # x shape: (batch, channels, temporal, height, width) or (batch, channels, height, width)
        if len(x.shape) == 5:
            # Video format: (batch, channels, temporal, height, width)
            b, c, t, h, w = x.shape
            if h != self.expected_height or w != self.expected_width:
                # Resize: squeeze temporal, interpolate, unsqueeze back
                # This works for t=1, for multiple frames we'd need to loop
                if t == 1:
                    x_squeezed = x.squeeze(2)  # (b, c, h, w)
                    x_resized = torch.nn.functional.interpolate(
                        x_squeezed, size=(self.expected_height, self.expected_width),
                        mode='bilinear', align_corners=False
                    )
                    x = x_resized.unsqueeze(2)  # (b, c, 1, h', w')
                else:
                    # For multiple frames, process each frame
                    frames = []
                    for i in range(t):
                        frame = x[:, :, i, :, :]  # (b, c, h, w)
                        frame_resized = torch.nn.functional.interpolate(
                            frame, size=(self.expected_height, self.expected_width),
                            mode='bilinear', align_corners=False
                        )
                        frames.append(frame_resized)
                    x = torch.stack(frames, dim=2)  # (b, c, t, h', w')
        elif len(x.shape) == 4:
            # Image format: (batch, channels, height, width)
            b, c, h, w = x.shape
            if h != self.expected_height or w != self.expected_width:
                x = torch.nn.functional.interpolate(
                    x, size=(self.expected_height, self.expected_width),
                    mode='bilinear', align_corners=False
                )

        # Run encoder on Neuron
        h = self.compiled_encoder(x)

        # Apply quant_conv if compiled
        if self.compiled_quant_conv is not None:
            moments = self.compiled_quant_conv(h)
        else:
            moments = h

        # Split into mean and logvar
        mean, logvar = moments.chunk(2, dim=1)

        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(std)

        # Scale latents
        sample = (sample - self.latents_mean) / self.latents_std

        # Scale the mean as well for mode()
        scaled_mean = (mean - self.latents_mean) / self.latents_std

        if return_dict:
            # Create a proper class for latent distribution
            class LatentDist:
                def __init__(self, sample_val, mean_val):
                    self._sample = sample_val
                    self._mean = mean_val
                def sample(self):
                    return self._sample
                def mode(self):
                    return self._mean
                @property
                def mean(self):
                    return self._mean

            class EncoderOutput:
                def __init__(self, latent_dist):
                    self.latent_dist = latent_dist

            return EncoderOutput(LatentDist(sample, scaled_mean))
        return sample

    def decode(self, z, return_dict=True):
        """Decode latents to images on Neuron."""
        # Unscale latents
        z = z * self.latents_std + self.latents_mean

        # Apply post_quant_conv if compiled
        if self.compiled_post_quant_conv is not None:
            z = self.compiled_post_quant_conv(z)

        # Run decoder on Neuron
        dec = self.compiled_decoder(z)

        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=dec)
        return dec


def load_all_compiled_models(compiled_models_dir: str, pipe, args):
    """
    Load ALL compiled models for Trainium2 inference.
    Every component MUST be compiled and loaded.

    Parallel configuration:
    - VAE: DataParallel (DP=8) - single-device compiled, replicated across 8 devices
    - Transformer: Tensor Parallel (TP=8) - sharded across 8 devices
    - Vision Encoder: DataParallel (DP=8) - single-device compiled, replicated
    - Language Model: Tensor Parallel (TP=4) - sharded across 4 devices

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Original pipeline
        args: Command line arguments

    Returns:
        Updated pipeline with ALL Neuron-compiled models
    """
    print("\n" + "=" * 60)
    print("Loading Compiled Models for Trainium2")
    print("=" * 60)
    print("Parallel configuration:")
    print("  - VAE: DP=8")
    print("  - Transformer: TP=8")
    print("  - Vision Encoder: DP=8")
    print("  - Language Model: TP=4")

    # ========================================
    # 1. Load Text Encoder Components
    # ========================================
    print("\n[1/3] Loading Text Encoder...")

    # Load Vision Encoder
    vision_encoder_path = f"{compiled_models_dir}/vision_encoder/model.pt"
    if not os.path.exists(vision_encoder_path):
        raise FileNotFoundError(
            f"Vision encoder not found at {vision_encoder_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_text_encoder.py --vision_only"
        )
    print(f"  Loading vision encoder from {vision_encoder_path}...")
    # Vision encoder uses [num_patches, channels] input (not batch), so no DataParallel
    compiled_vision_encoder = torch.jit.load(vision_encoder_path)
    print("  Vision encoder loaded!")

    # Load Language Model (TP=4)
    language_model_path = f"{compiled_models_dir}/language_model"
    if not os.path.exists(language_model_path):
        raise FileNotFoundError(
            f"Language model not found at {language_model_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_text_encoder.py --language_only"
        )
    print(f"  Loading language model from {language_model_path}...")
    compiled_language_model = neuronx_distributed.trace.parallel_model_load(
        language_model_path
    )
    print("  Language model loaded (TP=4)!")

    # Create Text Encoder Wrapper
    pipe.text_encoder = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder=compiled_vision_encoder,
        compiled_language_model=compiled_language_model,
        image_size=args.image_size,
        max_seq_len=args.max_sequence_length
    )
    print("  Text encoder wrapper created!")

    # ========================================
    # 2. Load Transformer
    # ========================================
    print("\n[2/3] Loading Transformer...")

    transformer_path = f"{compiled_models_dir}/transformer"
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(
            f"Transformer not found at {transformer_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer.py"
        )
    print(f"  Loading transformer from {transformer_path}...")
    compiled_transformer = neuronx_distributed.trace.parallel_model_load(
        transformer_path
    )

    # Calculate img_shapes for the compiled model
    latent_h = args.height // 8
    latent_w = args.width // 8
    patch_h = latent_h // 2
    patch_w = latent_w // 2
    # batch_size: 2 for CFG (conditional + unconditional), 1 for no CFG
    img_shapes = [(1, patch_h, patch_w)] * args.transformer_batch_size

    pipe.transformer = NeuronTransformerWrapper(
        pipe.transformer, compiled_transformer, img_shapes
    )
    print("  Transformer loaded (TP=8)!")

    # ========================================
    # 3. Load VAE (Encoder + Decoder)
    # ========================================
    print("\n[3/3] Loading VAE...")

    # First replace with Neuron-compatible VAE architecture
    print("  Creating Neuron-compatible VAE...")
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

    # Load compiled encoder
    vae_encoder_path = f"{compiled_models_dir}/vae_encoder/model.pt"
    if not os.path.exists(vae_encoder_path):
        raise FileNotFoundError(
            f"VAE encoder not found at {vae_encoder_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_vae.py"
        )
    print(f"  Loading VAE encoder from {vae_encoder_path}...")
    vae_encoder_jit = torch.jit.load(vae_encoder_path)
    # Wrap with DataParallel (DP=8)
    compiled_encoder = torch_neuronx.DataParallel(
        vae_encoder_jit,
        device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        set_dynamic_batching=False
    )
    print("  VAE encoder loaded (DP=8)!")

    # Load compiled decoder
    vae_decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    if not os.path.exists(vae_decoder_path):
        raise FileNotFoundError(
            f"VAE decoder not found at {vae_decoder_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_vae.py"
        )
    print(f"  Loading VAE decoder from {vae_decoder_path}...")
    vae_decoder_jit = torch.jit.load(vae_decoder_path)
    # Wrap with DataParallel (DP=8)
    compiled_decoder = torch_neuronx.DataParallel(
        vae_decoder_jit,
        device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        set_dynamic_batching=False
    )
    print("  VAE decoder loaded (DP=8)!")

    # Load quant_conv and post_quant_conv if they exist (also with DP=8)
    compiled_quant_conv = None
    quant_conv_path = f"{compiled_models_dir}/quant_conv/model.pt"
    if os.path.exists(quant_conv_path):
        print(f"  Loading quant_conv from {quant_conv_path}...")
        quant_conv_jit = torch.jit.load(quant_conv_path)
        compiled_quant_conv = torch_neuronx.DataParallel(
            quant_conv_jit,
            device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            set_dynamic_batching=False
        )

    compiled_post_quant_conv = None
    post_quant_conv_path = f"{compiled_models_dir}/post_quant_conv/model.pt"
    if os.path.exists(post_quant_conv_path):
        print(f"  Loading post_quant_conv from {post_quant_conv_path}...")
        post_quant_conv_jit = torch.jit.load(post_quant_conv_path)
        compiled_post_quant_conv = torch_neuronx.DataParallel(
            post_quant_conv_jit,
            device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            set_dynamic_batching=False
        )

    # Create VAE Wrapper
    pipe.vae = NeuronVAEWrapper(
        original_vae=neuron_vae,
        compiled_encoder=compiled_encoder,
        compiled_decoder=compiled_decoder,
        compiled_quant_conv=compiled_quant_conv,
        compiled_post_quant_conv=compiled_post_quant_conv,
        expected_height=args.height,
        expected_width=args.width
    )
    print("  VAE wrapper created!")

    # Fix missing _execution_device property
    # The pipeline expects this to determine where to run operations
    # Override the property with a lambda that returns CPU device
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))

    print("\n" + "=" * 60)
    print("All Models Loaded on Trainium2!")
    print("=" * 60)

    return pipe


def run_inference(args):
    """Run image editing inference on Trainium2."""
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("Qwen-Image-Edit Inference on Trainium2")
    print("=" * 60)
    print(f"  Height: {args.height}")
    print(f"  Width: {args.width}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    dtype = torch.bfloat16

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=HUGGINGFACE_CACHE_DIR,
        local_files_only=True
    )
    print("Pipeline loaded!")

    # Load ALL compiled models - everything runs on Trainium2
    pipe = load_all_compiled_models(args.compiled_models_dir, pipe, args)

    # Load source images (1-3 images supported)
    print(f"\nLoading {len(args.images)} source image(s)...")
    source_images = []
    for img_path in args.images:
        print(f"  Loading: {img_path}")
        img = load_image(img_path)
        # Resize to match compiled dimensions
        img = img.resize((args.width, args.height))
        source_images.append(img)
    print(f"All images resized to: {args.width}x{args.height}")

    # Use single image or list based on count
    input_images = source_images[0] if len(source_images) == 1 else source_images

    # Create generator for reproducibility
    generator = torch.Generator().manual_seed(args.seed)

    # Handle CFG based on transformer_batch_size
    guidance_scale = args.guidance_scale
    if args.transformer_batch_size == 1:
        if args.guidance_scale != 1.0:
            print(f"  WARNING: transformer_batch_size=1, forcing guidance_scale=1.0 (no CFG)")
        guidance_scale = 1.0  # CFG requires batch_size=2

    # Warmup run
    if args.warmup:
        print("\n" + "-" * 40)
        print("Running warmup inference...")
        print("-" * 40)
        warmup_generator = torch.Generator().manual_seed(args.seed + 1000)
        start = time.time()
        _ = pipe(
            image=input_images,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            guidance_scale=guidance_scale,
            num_inference_steps=min(5, args.num_inference_steps),
            generator=warmup_generator,
        )
        warmup_time = time.time() - start
        print(f"Warmup time: {warmup_time:.2f}s")

    # Main inference
    print("\n" + "-" * 40)
    print("Running main inference...")
    print("-" * 40)
    print(f"  Prompt: {args.prompt}")

    generator = torch.Generator().manual_seed(args.seed)
    start = time.time()
    output = pipe(
        image=input_images,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        guidance_scale=guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    inference_time = time.time() - start

    print(f"\nInference time: {inference_time:.2f}s")

    # Save output
    output_image = output.images[0]
    output_path = args.output or "output_edited.png"
    output_image.save(output_path)
    print(f"Output saved to: {output_path}")

    # Save comparison
    if args.save_comparison:
        # Create comparison with all input images + output
        num_images = len(source_images) + 1  # inputs + output
        comparison = Image.new('RGB', (args.width * num_images, args.height))
        for i, img in enumerate(source_images):
            comparison.paste(img, (args.width * i, 0))
        comparison.paste(output_image, (args.width * len(source_images), 0))
        comparison_path = output_path.replace('.png', '_comparison.png')
        comparison.save(comparison_path)
        print(f"Comparison saved to: {comparison_path}")

    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit inference on AWS Trainium2 (ALL components on Neuron)"
    )

    # Input/Output
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Path(s) to source image(s) for editing (1-3 images supported)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Edit instruction prompt")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: output_edited.png)")

    # Image settings - MUST match compilation settings
    parser.add_argument("--height", type=int, default=512,
                        help="Image height (must match compiled model)")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (must match compiled model)")

    # Text encoder settings - MUST match compilation settings
    parser.add_argument("--image_size", type=int, default=224,
                        help="Vision encoder image size (must match compiled model)")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length (must match compiled model)")
    parser.add_argument("--transformer_batch_size", type=int, default=1,
                        help="Transformer batch size (1=no CFG, 2=with CFG, must match compiled model)")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")

    # Model settings
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")

    # Other options
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup inference before main inference")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save side-by-side comparison image")

    args = parser.parse_args()

    # Validate number of images (1-3 supported by Qwen-Image-Edit)
    if len(args.images) > 3:
        parser.error("Qwen-Image-Edit supports 1-3 images, but {} were provided".format(len(args.images)))

    run_inference(args)
