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

# ============================================================================
# CRITICAL: Set Neuron environment variables BEFORE any other imports!
# These MUST match the compilation settings.
# ============================================================================
# NOTE: Transformer uses TP=8. Language Model can run on:
# - Neuron with TP=4 (correct GQA alignment, but requires separate process)
# - CPU (slower but works in same process as TP=8 Transformer)
#
# GQA alignment issue: 28Q/4KV heads requires TP=4 for correct alignment,
# but TP=4 causes OOM on Transformer. So we default to CPU Language Model.
TP_DEGREE = 8  # For Transformer; Language Model runs on CPU by default

# Set tensor parallel world size
os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)

# Neuron runtime settings - MUST match compilation
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # For trn2 LNC=2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"     # For trn2 LNC=2

# Neuron compiler settings (for any runtime compilation)
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

print(f"Neuron runtime configured: TP={TP_DEGREE}, LNC=2")

import argparse
import contextlib
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
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
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
    def __init__(self, original_transformer, compiled_transformer, img_shapes,
                 expected_num_patches=1024, expected_seq_len=512):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.compiled_transformer = compiled_transformer
        self.img_shapes = img_shapes
        self.expected_num_patches = expected_num_patches
        self.expected_seq_len = expected_seq_len

    @contextlib.contextmanager
    def cache_context(self, name: str):
        """Dummy cache context for compatibility with pipeline.
        Compiled models don't use dynamic caching."""
        yield

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, img_shapes=None, return_dict=False, **kwargs):
        """
        Forward pass using compiled transformer on Neuron.
        Handles shape padding and dtype conversion for compiled model.
        """
        batch_size = hidden_states.shape[0]

        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG Transformer input shapes:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"  timestep: {timestep.shape}, dtype={timestep.dtype}")
            print(f"  img_shapes: {img_shapes}")
            print(f"  Expected: num_patches={self.expected_num_patches}, seq_len={self.expected_seq_len}")
            self._debug_printed = True

        # 1. Handle hidden_states shape (num_patches dimension)
        # Compiled model expects (batch, expected_num_patches, 64)
        actual_patches = hidden_states.shape[1]
        if actual_patches != self.expected_num_patches:
            if actual_patches < self.expected_num_patches:
                # Pad with zeros
                pad_size = self.expected_num_patches - actual_patches
                padding = torch.zeros(
                    (batch_size, pad_size, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            else:
                # Truncate - This is problematic! The model was compiled for fewer patches.
                # This likely means the transformer needs to be recompiled with correct shape.
                print(f"ERROR: hidden_states has {actual_patches} patches but model expects {self.expected_num_patches}")
                print(f"  You may need to recompile the transformer with correct dimensions.")
                print(f"  Truncating will produce incorrect results!")
                hidden_states = hidden_states[:, :self.expected_num_patches, :]

        # 2. Handle encoder_hidden_states shape (sequence length)
        # Compiled model expects (batch, expected_seq_len, 3584)
        actual_seq_len = encoder_hidden_states.shape[1]
        if actual_seq_len != self.expected_seq_len:
            if actual_seq_len < self.expected_seq_len:
                # Pad with zeros
                pad_size = self.expected_seq_len - actual_seq_len
                padding = torch.zeros(
                    (batch_size, pad_size, encoder_hidden_states.shape[2]),
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
            else:
                # Truncate
                print(f"WARNING: Truncating encoder_hidden_states from {actual_seq_len} to {self.expected_seq_len}")
                encoder_hidden_states = encoder_hidden_states[:, :self.expected_seq_len, :]

        # 3. Convert timestep to float32 (compiled model expects float32)
        timestep = timestep.to(torch.float32)

        # Run on compiled Neuron model
        output = self.compiled_transformer(
            hidden_states,
            encoder_hidden_states,
            timestep
        )

        # 4. Remove padding from output if we padded hidden_states
        if actual_patches < self.expected_num_patches:
            output = (output[0][:, :actual_patches, :],) + output[1:]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output[0])
        return output


class NeuronVAEWrapper(torch.nn.Module):
    """
    Wrapper for VAE with compiled encoder and decoder on Trainium2.

    Supports tiled processing for images larger than the compiled tile size.
    """
    def __init__(self, original_vae, compiled_encoder, compiled_decoder,
                 compiled_quant_conv=None, compiled_post_quant_conv=None,
                 expected_height=512, expected_width=512,
                 cpu_decode=False):
        super().__init__()
        self.config = original_vae.config
        self.dtype = original_vae.dtype

        # Compiled models - ALL run on Neuron
        self.compiled_encoder = compiled_encoder
        self.compiled_decoder = compiled_decoder
        self.compiled_quant_conv = compiled_quant_conv
        self.compiled_post_quant_conv = compiled_post_quant_conv

        # CPU decode mode for debugging
        self.cpu_decode = cpu_decode
        if cpu_decode:
            print("  [DEBUG] VAE Decoder will run on CPU!")
            # Keep CPU decoder and post_quant_conv
            self.cpu_decoder = original_vae.decoder
            self.cpu_post_quant_conv = original_vae.post_quant_conv
            self.cpu_decoder.eval()

        # Scaling factors - convert to tensors for broadcasting
        # Shape: (1, z_dim, 1, 1, 1) for proper broadcasting with 5D latents (b, c, t, h, w)
        if isinstance(original_vae.latents_mean, list):
            self.latents_mean = torch.tensor(original_vae.latents_mean).view(1, -1, 1, 1, 1)
        else:
            self.latents_mean = original_vae.latents_mean
        if isinstance(original_vae.latents_std, list):
            self.latents_std = torch.tensor(original_vae.latents_std).view(1, -1, 1, 1, 1)
        else:
            self.latents_std = original_vae.latents_std

        # z_dim for shape calculations
        self.z_dim = original_vae.config.z_dim

        # Expected input size for compiled model (tile size)
        self.expected_height = expected_height
        self.expected_width = expected_width

        # Tiling parameters for larger images
        self.tile_sample_min_height = expected_height
        self.tile_sample_min_width = expected_width
        # Overlap between tiles (for blending)
        self.tile_overlap = 64  # pixels of overlap
        self.tile_sample_stride_height = expected_height - self.tile_overlap
        self.tile_sample_stride_width = expected_width - self.tile_overlap
        # Spatial compression ratio (8x for this VAE)
        self.spatial_compression_ratio = 8

    def _needs_tiling(self, h, w):
        """Check if image needs tiled processing."""
        return h > self.expected_height or w > self.expected_width

    def _blend_v(self, a, b, blend_extent):
        """Blend two tensors vertically."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def _blend_h(self, a, b, blend_extent):
        """Blend two tensors horizontally."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def _encode_tile(self, x):
        """Encode a single tile through compiled encoder."""
        h = self.compiled_encoder(x)
        if self.compiled_quant_conv is not None:
            moments = self.compiled_quant_conv(h)
        else:
            moments = h
        return moments

    def _decode_tile(self, z):
        """Decode a single tile through compiled decoder."""
        if self.compiled_post_quant_conv is not None:
            z = self.compiled_post_quant_conv(z)
        return self.compiled_decoder(z)

    def encode(self, x, return_dict=True):
        """Encode images to latents on Neuron. Supports tiled encoding for large images."""
        # Ensure 5D format: (batch, channels, temporal, height, width)
        if len(x.shape) == 4:
            x = x.unsqueeze(2)  # Add temporal dimension

        b, c, t, h, w = x.shape

        # Convert to bfloat16 (compiled models expect bfloat16)
        x = x.to(torch.bfloat16)

        # Check if tiling is needed
        if self._needs_tiling(h, w):
            print(f"  Using tiled encoding: {h}x{w} -> tiles of {self.expected_height}x{self.expected_width}")
            moments = self._tiled_encode(x)
        else:
            # Pad to expected size if smaller
            if h != self.expected_height or w != self.expected_width:
                # Pad with zeros
                pad_h = self.expected_height - h
                pad_w = self.expected_width - w
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

            moments = self._encode_tile(x)

            # Remove padding from latents if we padded
            if h != self.expected_height or w != self.expected_width:
                latent_h = h // self.spatial_compression_ratio
                latent_w = w // self.spatial_compression_ratio
                moments = moments[:, :, :, :latent_h, :latent_w]

        # Split into mean and logvar
        mean, logvar = moments.chunk(2, dim=1)

        # Sample from distribution (for sample() method)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(std)

        if return_dict:
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

            return EncoderOutput(LatentDist(sample, mean))
        return sample

    def _tiled_encode(self, x):
        """Encode large image using tiled processing."""
        b, c, t, h, w = x.shape

        # Latent dimensions
        latent_h = h // self.spatial_compression_ratio
        latent_w = w // self.spatial_compression_ratio
        tile_latent_h = self.expected_height // self.spatial_compression_ratio
        tile_latent_w = self.expected_width // self.spatial_compression_ratio
        tile_latent_stride_h = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_w = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_h = tile_latent_h - tile_latent_stride_h
        blend_w = tile_latent_w - tile_latent_stride_w

        # Process tiles
        rows = []
        for i in range(0, h, self.tile_sample_stride_height):
            row = []
            for j in range(0, w, self.tile_sample_stride_width):
                # Extract tile (with padding if at edge)
                tile_h_end = min(i + self.tile_sample_min_height, h)
                tile_w_end = min(j + self.tile_sample_min_width, w)
                tile = x[:, :, :, i:tile_h_end, j:tile_w_end]

                # Pad tile to expected size if needed
                actual_h, actual_w = tile.shape[3], tile.shape[4]
                if actual_h < self.expected_height or actual_w < self.expected_width:
                    pad_h = self.expected_height - actual_h
                    pad_w = self.expected_width - actual_w
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))

                # Encode tile
                encoded_tile = self._encode_tile(tile)

                # Crop encoded tile if we padded
                if actual_h < self.expected_height or actual_w < self.expected_width:
                    crop_h = actual_h // self.spatial_compression_ratio
                    crop_w = actual_w // self.spatial_compression_ratio
                    encoded_tile = encoded_tile[:, :, :, :crop_h, :crop_w]

                row.append(encoded_tile)
            rows.append(row)

        # Blend tiles together
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :tile_latent_stride_h, :tile_latent_stride_w])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=3)[:, :, :, :latent_h, :latent_w]

    def decode(self, z, return_dict=True):
        """Decode latents to images on Neuron. Supports tiled decoding for large latents."""
        # NOTE: Do NOT unscale latents here!
        # The pipeline already unscales latents before calling decode

        # Ensure 5D format
        if len(z.shape) == 4:
            z = z.unsqueeze(2)

        b, c, t, latent_h, latent_w = z.shape

        # Convert to bfloat16
        z = z.to(torch.bfloat16)

        # Calculate output image size
        output_h = latent_h * self.spatial_compression_ratio
        output_w = latent_w * self.spatial_compression_ratio

        if self.cpu_decode:
            # CPU decode mode for debugging
            z_cpu = z.to(torch.float32)
            with torch.no_grad():
                z_cpu = self.cpu_post_quant_conv(z_cpu)
                dec = self.cpu_decoder(z_cpu)
            dec = dec.to(torch.bfloat16)
        elif self._needs_tiling(output_h, output_w):
            print(f"  Using tiled decoding: latent {latent_h}x{latent_w} -> image {output_h}x{output_w}")
            dec = self._tiled_decode(z)
        else:
            # Check if latent needs padding to match compiled size
            expected_latent_h = self.expected_height // self.spatial_compression_ratio
            expected_latent_w = self.expected_width // self.spatial_compression_ratio

            if latent_h != expected_latent_h or latent_w != expected_latent_w:
                # Pad latents
                pad_h = expected_latent_h - latent_h
                pad_w = expected_latent_w - latent_w
                z = torch.nn.functional.pad(z, (0, pad_w, 0, pad_h))

            dec = self._decode_tile(z)

            # Crop output if we padded
            if latent_h != expected_latent_h or latent_w != expected_latent_w:
                dec = dec[:, :, :, :output_h, :output_w]

        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=dec)
        return (dec,)

    def _tiled_decode(self, z):
        """Decode large latents using tiled processing."""
        b, c, t, latent_h, latent_w = z.shape

        # Calculate dimensions
        output_h = latent_h * self.spatial_compression_ratio
        output_w = latent_w * self.spatial_compression_ratio

        tile_latent_h = self.expected_height // self.spatial_compression_ratio
        tile_latent_w = self.expected_width // self.spatial_compression_ratio
        tile_latent_stride_h = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_w = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_w = self.tile_sample_min_width - self.tile_sample_stride_width

        # Process tiles
        rows = []
        for i in range(0, latent_h, tile_latent_stride_h):
            row = []
            for j in range(0, latent_w, tile_latent_stride_w):
                # Extract latent tile (with padding if at edge)
                tile_h_end = min(i + tile_latent_h, latent_h)
                tile_w_end = min(j + tile_latent_w, latent_w)
                tile = z[:, :, :, i:tile_h_end, j:tile_w_end]

                # Pad tile to expected size if needed
                actual_h, actual_w = tile.shape[3], tile.shape[4]
                if actual_h < tile_latent_h or actual_w < tile_latent_w:
                    pad_h = tile_latent_h - actual_h
                    pad_w = tile_latent_w - actual_w
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))

                # Decode tile
                decoded_tile = self._decode_tile(tile)

                # Crop decoded tile if we padded
                if actual_h < tile_latent_h or actual_w < tile_latent_w:
                    crop_h = actual_h * self.spatial_compression_ratio
                    crop_w = actual_w * self.spatial_compression_ratio
                    decoded_tile = decoded_tile[:, :, :, :crop_h, :crop_w]

                row.append(decoded_tile)
            rows.append(row)

        # Blend tiles together
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :self.tile_sample_stride_height, :self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=3)[:, :, :, :output_h, :output_w]


def load_all_compiled_models(compiled_models_dir: str, pipe, args):
    """
    Load ALL compiled models for Trainium2 inference.
    Every component MUST be compiled and loaded.

    Parallel configuration:
    - VAE: DataParallel (DP=8) - single-device compiled, replicated across 8 devices
    - Transformer: Tensor Parallel (TP=8) - sharded across 8 devices
    - Vision Encoder: Single device OR TP=8 (use --vision_tp flag for TP mode)
    - Language Model: Tensor Parallel (TP=8) - sharded with KV head replication

    IMPORTANT: This function replaces original models with compiled versions
    and explicitly deletes the originals to free memory.

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Original pipeline
        args: Command line arguments

    Returns:
        Updated pipeline with ALL Neuron-compiled models
    """
    import gc

    # Check for vision encoder mode
    # --neuron_vision_encoder overrides default --cpu_vision_encoder
    vision_encoder_tp_path = f"{compiled_models_dir}/vision_encoder_tp"
    use_vision_tp = args.vision_tp if hasattr(args, 'vision_tp') else False
    use_neuron_vision = getattr(args, 'neuron_vision_encoder', False)
    use_cpu_vision_encoder = not use_neuron_vision  # Default to CPU unless --neuron_vision_encoder
    if use_cpu_vision_encoder:
        vision_mode = "CPU (highest accuracy, default)"
    elif use_vision_tp or os.path.exists(vision_encoder_tp_path):
        vision_mode = "TP=8"
    else:
        vision_mode = "single device"

    print("\n" + "=" * 60)
    print("Loading Compiled Models for Trainium2")
    print("=" * 60)
    # Check if using CPU language model
    # --neuron_language_model overrides --cpu_language_model
    use_cpu_language_model = not getattr(args, 'neuron_language_model', False)
    language_mode = "CPU" if use_cpu_language_model else "Neuron (compiled)"

    print("Parallel configuration:")
    print("  - VAE: Single device (avoid collective conflict)")
    print("  - Transformer: TP=8")
    print(f"  - Vision Encoder: {vision_mode}")
    print(f"  - Language Model: {language_mode}")
    if use_cpu_language_model:
        print("\nNOTE: Language Model on CPU due to GQA alignment issue with TP=8")
        print("      (28 Q heads / 4 KV heads requires TP=4, which causes OOM on Transformer)")
    else:
        print("\nNOTE: All TP models use TP=8 for consistent world_size")

    # ========================================
    # 1. Load Transformer FIRST (TP=8)
    # ========================================
    # IMPORTANT: Must load the largest TP model first to initialize
    # the communicator with the correct world size
    print("\n[1/3] Loading Transformer (TP=8)...")

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

    # Calculate expected shapes for the COMPILED model
    # These must match what was used during compilation (compile.sh defaults to 512x512)
    compiled_height = args.compiled_height
    compiled_width = args.compiled_width
    compiled_latent_h = compiled_height // 8
    compiled_latent_w = compiled_width // 8
    compiled_patch_h = compiled_latent_h // 2
    compiled_patch_w = compiled_latent_w // 2
    base_num_patches = compiled_patch_h * compiled_patch_w  # 32*32=1024 for 512x512

    # For IMAGE EDITING, patches are doubled (source + noise latents concatenated)
    # This is handled by using temporal_frames = patch_multiplier
    # - patch_multiplier=1 (generation): temporal_frames=1, patches = 1 * 32 * 32 = 1024
    # - patch_multiplier=2 (editing): temporal_frames=2, patches = 2 * 32 * 32 = 2048
    temporal_frames = args.patch_multiplier
    expected_num_patches = temporal_frames * base_num_patches
    print(f"  Expected num_patches: {expected_num_patches} (temporal_frames={temporal_frames}, base={base_num_patches})")

    # img_shapes for the wrapper - must match compiled model
    # Use compiled dimensions and temporal_frames = patch_multiplier
    patch_h = compiled_patch_h
    patch_w = compiled_patch_w
    img_shapes = [(temporal_frames, patch_h, patch_w)] * args.transformer_batch_size

    # Store reference to original for wrapper, then delete
    original_transformer = pipe.transformer
    pipe.transformer = NeuronTransformerWrapper(
        original_transformer, compiled_transformer, img_shapes,
        expected_num_patches=expected_num_patches,
        expected_seq_len=args.max_sequence_length
    )
    # Delete original transformer to free ~40GB memory
    del original_transformer
    gc.collect()
    print(f"  Transformer loaded (TP=8)! Expected patches={expected_num_patches}, seq_len={args.max_sequence_length}")
    print("  Original transformer deleted to free memory.")

    # ========================================
    # 2. Load Text Encoder Components
    # ========================================
    print("\n[2/3] Loading Text Encoder...")

    # Load Vision Encoder
    # Check for TP version first (better memory distribution), then single device
    # Note: vision_encoder_tp_path, use_vision_tp, use_cpu_vision_encoder are defined at the top
    vision_encoder_single_path = f"{compiled_models_dir}/vision_encoder/model.pt"
    compiled_vision_encoder = None
    cpu_vision_encoder = None

    if use_cpu_vision_encoder:
        # CPU Vision Encoder mode - highest accuracy, avoids compilation precision loss
        # This is useful when compiled vision encoder produces blurry outputs
        print("  Using CPU Vision Encoder (highest accuracy)...")
        # Extract vision encoder from text encoder - will be passed to wrapper
        cpu_vision_encoder = pipe.text_encoder.model.visual
        cpu_vision_encoder.eval()
        print("  Vision encoder prepared on CPU!")
    elif use_vision_tp or (os.path.exists(vision_encoder_tp_path) and not os.path.exists(vision_encoder_single_path)):
        # Load TP-compiled vision encoder
        if not os.path.exists(vision_encoder_tp_path):
            raise FileNotFoundError(
                f"Vision encoder (TP) not found at {vision_encoder_tp_path}\n"
                "Please run: python neuron_qwen_image_edit/compile_text_encoder.py --vision_only --vision_tp"
            )
        print(f"  Loading vision encoder (TP={TP_DEGREE}) from {vision_encoder_tp_path}...")
        compiled_vision_encoder = neuronx_distributed.trace.parallel_model_load(
            vision_encoder_tp_path
        )
        print(f"  Vision encoder loaded (TP={TP_DEGREE})!")
    else:
        # Load single-device vision encoder
        if not os.path.exists(vision_encoder_single_path):
            raise FileNotFoundError(
                f"Vision encoder not found at {vision_encoder_single_path}\n"
                "Please run: python neuron_qwen_image_edit/compile_text_encoder.py --vision_only\n"
                "Or for TP version: python neuron_qwen_image_edit/compile_text_encoder.py --vision_only --vision_tp"
            )
        print(f"  Loading vision encoder from {vision_encoder_single_path}...")
        vision_encoder_jit = torch.jit.load(vision_encoder_single_path)
        # Vision encoder input is (num_patches, channels), NOT (batch, ...)
        # DataParallel would incorrectly split on patches dimension
        # Must use single device
        compiled_vision_encoder = vision_encoder_jit
        print("  Vision encoder loaded (single device - input is patches, not batch)!")

    # Load Language Model
    compiled_language_model = None
    cpu_language_model = None

    if use_cpu_language_model:
        # CPU Language Model mode - keeps original model on CPU
        # This avoids GQA alignment issues that occur with TP != 4
        print("  Using CPU Language Model (avoids GQA alignment issue)...")
        # Extract language model from text encoder BEFORE creating wrapper
        cpu_language_model = pipe.text_encoder.model.language_model
        cpu_language_model.eval()
        # Keep it in bfloat16 for memory efficiency
        cpu_language_model = cpu_language_model.to(torch.bfloat16)
        print("  Language model prepared on CPU!")
    else:
        # Neuron compiled Language Model mode (TP=8 with KV head replication)
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
        print("  Language model loaded (TP=8 with KV head replication)!")

    # Create Text Encoder Wrapper
    # Store reference to original, then delete after wrapper is created
    original_text_encoder = pipe.text_encoder
    pipe.text_encoder = NeuronTextEncoderWrapper(
        original_text_encoder=original_text_encoder,
        compiled_vision_encoder=compiled_vision_encoder,
        compiled_language_model=compiled_language_model,
        cpu_language_model=cpu_language_model,
        cpu_vision_encoder=cpu_vision_encoder,
        image_size=args.image_size,
        max_seq_len=args.max_sequence_length
    )

    if use_cpu_language_model or use_cpu_vision_encoder:
        # When using CPU models, we keep references - don't delete original
        print("  Text encoder wrapper created!")
        if use_cpu_language_model:
            print("  Language model kept on CPU.")
        if use_cpu_vision_encoder:
            print("  Vision encoder kept on CPU (highest accuracy mode).")
    else:
        # Delete original text encoder to free ~16GB memory
        del original_text_encoder
        gc.collect()
        print("  Text encoder wrapper created!")
        print("  Original text encoder deleted to free memory.")

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
    # Use single device to avoid collective communication conflict with TP models
    # VAE is small (~300M params), doesn't need parallelism
    compiled_encoder = vae_encoder_jit
    print("  VAE encoder loaded (single device)!")

    # Load compiled decoder
    vae_decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    if not os.path.exists(vae_decoder_path):
        raise FileNotFoundError(
            f"VAE decoder not found at {vae_decoder_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_vae.py"
        )
    print(f"  Loading VAE decoder from {vae_decoder_path}...")
    vae_decoder_jit = torch.jit.load(vae_decoder_path)
    # Use single device to avoid collective communication conflict with TP models
    # VAE is small (~300M params), doesn't need parallelism
    compiled_decoder = vae_decoder_jit
    print("  VAE decoder loaded (single device)!")

    # Load quant_conv and post_quant_conv if they exist (single device)
    compiled_quant_conv = None
    quant_conv_path = f"{compiled_models_dir}/quant_conv/model.pt"
    if os.path.exists(quant_conv_path):
        print(f"  Loading quant_conv from {quant_conv_path}...")
        compiled_quant_conv = torch.jit.load(quant_conv_path)

    compiled_post_quant_conv = None
    post_quant_conv_path = f"{compiled_models_dir}/post_quant_conv/model.pt"
    if os.path.exists(post_quant_conv_path):
        print(f"  Loading post_quant_conv from {post_quant_conv_path}...")
        compiled_post_quant_conv = torch.jit.load(post_quant_conv_path)

    # Create VAE Wrapper
    cpu_decode = getattr(args, 'cpu_vae_decode', False)
    # Use vae_tile_size for the compiled model's expected input size
    vae_tile_size = getattr(args, 'vae_tile_size', 512)
    pipe.vae = NeuronVAEWrapper(
        original_vae=neuron_vae,
        compiled_encoder=compiled_encoder,
        compiled_decoder=compiled_decoder,
        compiled_quant_conv=compiled_quant_conv,
        compiled_post_quant_conv=compiled_post_quant_conv,
        expected_height=vae_tile_size,
        expected_width=vae_tile_size,
        cpu_decode=cpu_decode
    )
    # Delete the neuron_vae (original VAE copy) - small but still free it
    # Note: if cpu_decode=True, the decoder/post_quant_conv refs are already copied
    del neuron_vae
    gc.collect()
    print("  VAE wrapper created!")

    # Fix missing _execution_device property
    # The pipeline expects this to determine where to run operations
    # Override the property with a lambda that returns CPU device
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))

    # Use vision_mode and language_mode defined at the top of the function
    print("\n" + "=" * 60)
    print("All Models Loaded!")
    print("=" * 60)
    print("  - Transformer: Neuron (TP=8)")
    print(f"  - Language Model: {language_mode}")
    print(f"  - Vision Encoder: Neuron ({vision_mode})")
    print(f"  - VAE: Neuron (tile size={vae_tile_size}x{vae_tile_size})")
    print("")
    print("Tiled VAE note:")
    print(f"  - VAE compiled for {vae_tile_size}x{vae_tile_size} tiles")
    print("  - Larger images will be processed in tiles automatically")
    print("  - Example: 1024x1024 -> 4 tiles of 512x512 (with overlap)")
    print("")
    if use_cpu_language_model:
        print("Memory note:")
        print("  - Language Model on CPU (~8GB CPU memory)")
        print("  - Other components on Neuron")

    return pipe


def debug_text_encoder(pipe, input_images, args):
    """
    Debug: Compare NeuronTextEncoderWrapper output vs CPU.

    This function helps identify if text encoder is causing output issues.
    """
    import torch.nn.functional as F

    print("\nPreparing test input...")

    # Prepare input like the pipeline does
    prompt = args.prompt
    if isinstance(input_images, list):
        base_img_prompt = "".join([f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(len(input_images))])
        images = input_images
    else:
        base_img_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        images = [input_images]

    template = pipe.prompt_template_encode
    txt = [template.format(base_img_prompt + prompt)]

    model_inputs = pipe.processor(
        text=txt,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    print(f"  input_ids: {model_inputs.input_ids.shape}")
    print(f"  pixel_values: {model_inputs.pixel_values.shape}")
    print(f"  image_grid_thw: {model_inputs.image_grid_thw.tolist()}")

    # Count image tokens
    image_token_id = pipe.text_encoder.config.image_token_id if hasattr(pipe.text_encoder, 'config') else 151655
    num_image_tokens = (model_inputs.input_ids == image_token_id).sum().item()
    print(f"  Image tokens in input: {num_image_tokens}")

    # Run the wrapper (which is what inference uses)
    print("\nRunning NeuronTextEncoderWrapper...")
    with torch.no_grad():
        wrapper_output = pipe.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values.to(torch.bfloat16),
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

    if hasattr(wrapper_output, 'hidden_states'):
        wrapper_hidden = wrapper_output.hidden_states[-1]
    else:
        wrapper_hidden = wrapper_output.last_hidden_state

    print(f"  Wrapper output shape: {wrapper_hidden.shape}")
    print(f"  Wrapper output stats: mean={wrapper_hidden.float().mean():.4f}, std={wrapper_hidden.float().std():.4f}")
    print(f"  Wrapper output range: [{wrapper_hidden.float().min():.4f}, {wrapper_hidden.float().max():.4f}]")

    # Check for NaN/Inf
    has_nan = torch.isnan(wrapper_hidden).any().item()
    has_inf = torch.isinf(wrapper_hidden).any().item()
    if has_nan:
        print("  [WARNING] Output contains NaN!")
    if has_inf:
        print("  [WARNING] Output contains Inf!")

    # Save intermediate results for debugging
    debug_data = {
        'input_ids': model_inputs.input_ids.cpu().numpy(),
        'attention_mask': model_inputs.attention_mask.cpu().numpy(),
        'pixel_values_shape': list(model_inputs.pixel_values.shape),
        'image_grid_thw': model_inputs.image_grid_thw.cpu().numpy(),
        'wrapper_output': wrapper_hidden.float().cpu().numpy(),
    }

    import numpy as np
    np.savez('debug_text_encoder_output.npz', **debug_data)
    print("\n  Debug data saved to: debug_text_encoder_output.npz")
    print("  To compare with CPU, load original pipeline and run the same inputs.")


def run_inference(args):
    """Run image editing inference on Trainium2."""
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("Qwen-Image-Edit Inference on Trainium2")
    print("=" * 60)
    print(f"  Compiled dimensions: {args.compiled_height}x{args.compiled_width}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    dtype = torch.bfloat16

    # CRITICAL FIX: Override VAE_IMAGE_SIZE before loading pipeline
    # The pipeline uses VAE_IMAGE_SIZE (default 1024*1024) to resize source images.
    # This creates more patches than our compiled transformer expects.
    # We need to match our compiled dimensions.
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_pipeline_module
    compiled_vae_pixels = args.compiled_height * args.compiled_width  # e.g., 512*512
    original_vae_size = getattr(qwen_pipeline_module, 'VAE_IMAGE_SIZE', 1024*1024)
    qwen_pipeline_module.VAE_IMAGE_SIZE = compiled_vae_pixels
    print(f"\nOverriding VAE_IMAGE_SIZE: {original_vae_size} -> {compiled_vae_pixels}")
    print(f"  (This ensures source images produce {args.compiled_height//8//2}x{args.compiled_width//8//2} patches)")

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=HUGGINGFACE_CACHE_DIR,
        local_files_only=True
    )

    # CRITICAL: Configure processor to output fixed image size matching compiled vision encoder
    # The processor dynamically determines grid size based on min/max pixels.
    # We must force it to use the exact size the vision encoder was compiled for.
    target_pixels = args.image_size * args.image_size
    print(f"\nConfiguring processor for vision encoder size: {args.image_size}x{args.image_size}")
    print(f"  Setting min_pixels = max_pixels = {target_pixels}")
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels

    print("Pipeline loaded!")

    # Load ALL compiled models - everything runs on Trainium2
    pipe = load_all_compiled_models(args.compiled_models_dir, pipe, args)

    # Load source images (1-3 images supported)
    # IMPORTANT: Images must be resized to COMPILED dimensions for the transformer
    print(f"\nLoading {len(args.images)} source image(s)...")
    source_images = []
    for img_path in args.images:
        print(f"  Loading: {img_path}")
        img = load_image(img_path)
        # Resize to match COMPILED dimensions (not inference dimensions)
        img = img.resize((args.compiled_width, args.compiled_height))
        source_images.append(img)
    print(f"All images resized to: {args.compiled_width}x{args.compiled_height} (compiled dimensions)")

    # Use single image or list based on count
    input_images = source_images[0] if len(source_images) == 1 else source_images

    # Debug: Compare text encoder outputs
    if args.debug_text_encoder:
        print("\n" + "="*60)
        print("[DEBUG] Text Encoder Comparison")
        print("="*60)
        debug_text_encoder(pipe, input_images, args)
        print("="*60 + "\n")

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
            height=args.compiled_height,  # Use compiled dimensions
            width=args.compiled_width,
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
        height=args.compiled_height,  # Use compiled dimensions
        width=args.compiled_width,
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
        comparison = Image.new('RGB', (args.compiled_width * num_images, args.compiled_height))
        for i, img in enumerate(source_images):
            comparison.paste(img, (args.compiled_width * i, 0))
        comparison.paste(output_image, (args.compiled_width * len(source_images), 0))
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

    # Image settings for inference
    parser.add_argument("--height", type=int, default=512,
                        help="Image height for inference")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width for inference")

    # Compiled model dimensions (what the model was compiled with)
    parser.add_argument("--compiled_height", type=int, default=512,
                        help="Height used during model compilation (default: 512)")
    parser.add_argument("--compiled_width", type=int, default=512,
                        help="Width used during model compilation (default: 512)")
    parser.add_argument("--patch_multiplier", type=int, default=2,
                        help="Patch multiplier (2 for image editing, 1 for generation)")

    # Text encoder settings - MUST match compilation settings
    parser.add_argument("--image_size", type=int, default=224,
                        help="Vision encoder image size (must match compiled model)")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length (must match compiled model)")
    parser.add_argument("--transformer_batch_size", type=int, default=1,
                        help="Transformer batch size (1=no CFG, 2=with CFG, must match compiled model)")
    parser.add_argument("--vision_tp", action="store_true",
                        help="Use TP-compiled vision encoder (from vision_encoder_tp/). "
                             "Default is to auto-detect based on available compiled models.")

    # Language model mode
    parser.add_argument("--cpu_language_model", action="store_true", default=True,
                        help="Run Language Model on CPU (default). "
                             "This avoids GQA alignment issues with TP=8.")
    parser.add_argument("--neuron_language_model", action="store_true",
                        help="Use Neuron-compiled Language Model instead of CPU. "
                             "Requires compiled model with correct TP degree (usually TP=4).")

    # Vision encoder mode
    parser.add_argument("--cpu_vision_encoder", action="store_true", default=True,
                        help="Run Vision Encoder on CPU for higher accuracy (default). "
                             "Compiled Vision Encoder has precision loss that gets amplified by LM.")
    parser.add_argument("--neuron_vision_encoder", action="store_true",
                        help="Use Neuron-compiled Vision Encoder instead of CPU. "
                             "May have lower accuracy but faster speed.")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Number of denoising steps (default: 40)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")

    # Model settings
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--vae_tile_size", type=int, default=512,
                        help="VAE tile size (must match compiled VAE size). "
                             "For larger images, tiled VAE will process in this tile size.")

    # Other options
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup inference before main inference")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save side-by-side comparison image")

    # Debug options
    parser.add_argument("--cpu_vae_decode", action="store_true",
                        help="[DEBUG] Run VAE decoder on CPU instead of Neuron. "
                             "Use this to verify if other components are working correctly.")
    parser.add_argument("--debug_text_encoder", action="store_true",
                        help="[DEBUG] Compare Text Encoder outputs before running inference. "
                             "This helps identify if text encoder is the source of issues.")

    args = parser.parse_args()

    # Validate number of images (1-3 supported by Qwen-Image-Edit)
    if len(args.images) > 3:
        parser.error("Qwen-Image-Edit supports 1-3 images, but {} were provided".format(len(args.images)))

    run_inference(args)
