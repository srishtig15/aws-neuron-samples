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

# Import NxDModel for V2 API loading
try:
    from neuronx_distributed.trace.nxd_model.nxd_model import NxDModel
    NXD_MODEL_AVAILABLE = True
except ImportError:
    NXD_MODEL_AVAILABLE = False
    print("WARNING: NxDModel not available. V2 models cannot be loaded.")

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
        import time
        _t_start = time.time()
        output = self.compiled_transformer(
            hidden_states,
            encoder_hidden_states,
            timestep
        )
        # Profile transformer step (print every 10 steps to avoid spam)
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._step_times = []
        self._step_count += 1
        self._step_times.append(time.time() - _t_start)
        if self._step_count % 10 == 0:
            avg_time = sum(self._step_times[-10:]) / 10
            print(f"  [Profile] Transformer V1 (Neuron) step {self._step_count}: avg {avg_time:.3f}s/step")

        # 4. Remove padding from output if we padded hidden_states
        if actual_patches < self.expected_num_patches:
            output = (output[0][:, :actual_patches, :],) + output[1:]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output[0])
        return output


class NeuronTransformerWrapperV2(torch.nn.Module):
    """
    Wrapper for V2 compiled transformer (ModelBuilder API) on Trainium2.

    Key difference from V1: RoPE frequencies are passed as input, not computed internally.
    """
    def __init__(self, original_transformer, nxd_model, img_rotary_emb, txt_rotary_emb,
                 expected_num_patches=1024, expected_seq_len=512, temporal_frames=3):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        # Pre-computed RoPE frequencies
        self.img_rotary_emb = img_rotary_emb
        self.txt_rotary_emb = txt_rotary_emb

        self.expected_num_patches = expected_num_patches
        self.expected_seq_len = expected_seq_len
        self.temporal_frames = temporal_frames
        # Base patches per frame (noise prediction output size)
        self.base_patches = expected_num_patches // temporal_frames

    @contextlib.contextmanager
    def cache_context(self, name: str):
        """Dummy cache context for compatibility with pipeline."""
        yield

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, img_shapes=None, return_dict=False, **kwargs):
        """Forward pass using V2 compiled transformer with RoPE as input."""
        batch_size = hidden_states.shape[0]

        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG Transformer V2 input shapes:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"  timestep: {timestep.shape}")
            print(f"  img_rotary_emb: {self.img_rotary_emb.shape}")
            print(f"  txt_rotary_emb: {self.txt_rotary_emb.shape}")
            print(f"  temporal_frames: {self.temporal_frames}, base_patches: {self.base_patches}")
            print(f"  Will extract last {self.base_patches} patches as noise prediction")
            self._debug_printed = True

        # Handle hidden_states padding
        actual_patches = hidden_states.shape[1]
        if actual_patches != self.expected_num_patches:
            if actual_patches < self.expected_num_patches:
                pad_size = self.expected_num_patches - actual_patches
                padding = torch.zeros(
                    (batch_size, pad_size, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            else:
                print(f"ERROR: hidden_states has {actual_patches} patches but model expects {self.expected_num_patches}")
                hidden_states = hidden_states[:, :self.expected_num_patches, :]

        # Handle encoder_hidden_states padding
        actual_seq_len = encoder_hidden_states.shape[1]
        if actual_seq_len != self.expected_seq_len:
            if actual_seq_len < self.expected_seq_len:
                pad_size = self.expected_seq_len - actual_seq_len
                padding = torch.zeros(
                    (batch_size, pad_size, encoder_hidden_states.shape[2]),
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
            else:
                print(f"WARNING: Truncating encoder_hidden_states from {actual_seq_len} to {self.expected_seq_len}")
                encoder_hidden_states = encoder_hidden_states[:, :self.expected_seq_len, :]

        # Convert timestep to float32
        timestep = timestep.to(torch.float32)

        # Run V2 model with RoPE as input
        import time
        _t_start = time.time()
        output = self.nxd_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            self.img_rotary_emb,
            self.txt_rotary_emb
        )

        # Profile
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._step_times = []
        self._step_count += 1
        self._step_times.append(time.time() - _t_start)
        if self._step_count % 10 == 0:
            avg_time = sum(self._step_times[-10:]) / 10
            print(f"  [Profile] Transformer V2 (Neuron) step {self._step_count}: avg {avg_time:.3f}s/step")

        # Extract tensor from output (handle tuple or tensor)
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # For image editing, the model processes temporal_frames * base_patches
        # but should only return the noise prediction for one frame (base_patches)
        # Try extracting the FIRST frame (index 0) as noise prediction
        # (QwenImage may use frame 0 for noise, unlike other models that use last frame)
        output_tensor = output_tensor[:, :self.base_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


def load_transformer_v2(compiled_models_dir: str, pipe, args):
    """
    Load V2 compiled transformer model using NxDModel API.

    V2 models are compiled with ModelBuilder and require:
    1. nxd_model.pt - the compiled model
    2. weights/ - sharded checkpoints
    3. rope_cache.pt - pre-computed RoPE tensors
    4. config.json - model configuration

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Pipeline with original transformer (for config)
        args: Command line arguments

    Returns:
        NeuronTransformerWrapperV2 wrapping the loaded model
    """
    import json

    if not NXD_MODEL_AVAILABLE:
        raise RuntimeError(
            "NxDModel is not available. Please ensure neuronx_distributed is installed correctly."
        )

    v2_path = f"{compiled_models_dir}/transformer_v2"
    nxd_model_path = f"{v2_path}/nxd_model.pt"
    weights_path = f"{v2_path}/weights"
    rope_cache_path = f"{v2_path}/rope_cache.pt"
    config_path = f"{v2_path}/config.json"

    # Validate all required files exist
    if not os.path.exists(nxd_model_path):
        raise FileNotFoundError(
            f"V2 transformer model not found at {nxd_model_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2.py"
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"V2 transformer weights not found at {weights_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2.py"
        )
    if not os.path.exists(rope_cache_path):
        raise FileNotFoundError(
            f"V2 RoPE cache not found at {rope_cache_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2.py"
        )

    # Load config
    print(f"  Loading V2 config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    expected_num_patches = config["num_patches"]
    expected_seq_len = config["text_seq_len"]
    temporal_frames = config.get("frame", config.get("patch_multiplier", 3))
    base_patches = expected_num_patches // temporal_frames
    print(f"  V2 config: patches={expected_num_patches}, seq_len={expected_seq_len}")
    print(f"  V2 config: temporal_frames={temporal_frames}, base_patches={base_patches}")

    # Load pre-computed RoPE tensors
    print(f"  Loading RoPE cache from {rope_cache_path}...")
    rope_cache = torch.load(rope_cache_path)
    img_rotary_emb = rope_cache["img_rotary_emb"].to(torch.bfloat16)
    txt_rotary_emb = rope_cache["txt_rotary_emb"].to(torch.bfloat16)
    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    # Debug: Print RoPE statistics
    img_cos = img_rotary_emb[..., 0]
    img_sin = img_rotary_emb[..., 1]
    txt_cos = txt_rotary_emb[..., 0]
    txt_sin = txt_rotary_emb[..., 1]
    print(f"  img_cos stats: min={img_cos.min():.4f}, max={img_cos.max():.4f}, mean={img_cos.mean():.4f}")
    print(f"  img_sin stats: min={img_sin.min():.4f}, max={img_sin.max():.4f}, mean={img_sin.mean():.4f}")
    print(f"  txt_cos stats: min={txt_cos.min():.4f}, max={txt_cos.max():.4f}, mean={txt_cos.mean():.4f}")
    print(f"  txt_sin stats: min={txt_sin.min():.4f}, max={txt_sin.max():.4f}, mean={txt_sin.mean():.4f}")

    # Load the compiled model using NxDModel.load()
    print(f"  Loading V2 model from {nxd_model_path}...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded checkpoints
    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 8)
    print(f"  Loading sharded weights for TP={tp_degree}...")
    sharded_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_file(ckpt_path)
        sharded_checkpoints.append(ckpt)
        if rank == 0:
            print(f"    Rank 0 checkpoint keys: {len(ckpt)} tensors")

    # Initialize model with weights and move to Neuron
    print("  Setting weights...")
    nxd_model.set_weights(sharded_checkpoints)
    print("  Moving model to Neuron...")
    nxd_model.to_neuron()
    print("  V2 model initialized on Neuron!")

    # Create wrapper
    wrapper = NeuronTransformerWrapperV2(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        img_rotary_emb=img_rotary_emb,
        txt_rotary_emb=txt_rotary_emb,
        expected_num_patches=expected_num_patches,
        expected_seq_len=expected_seq_len,
        temporal_frames=temporal_frames,
    )

    return wrapper


class NeuronTransformerWrapperV1Flash(torch.nn.Module):
    """
    Wrapper for V1 Flash compiled transformer (parallel_model_trace + NKI Flash Attention).

    Key features:
    - Uses parallel_model_trace API (supports NKI Flash Attention)
    - RoPE frequencies are passed as input (like V2)
    - Uses NKI Flash Attention for better performance
    """
    def __init__(self, original_transformer, compiled_transformer, img_rotary_emb, txt_rotary_emb,
                 expected_num_patches=1024, expected_seq_len=512, temporal_frames=3):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.compiled_transformer = compiled_transformer

        # Pre-computed RoPE frequencies
        self.img_rotary_emb = img_rotary_emb
        self.txt_rotary_emb = txt_rotary_emb

        self.expected_num_patches = expected_num_patches
        self.expected_seq_len = expected_seq_len
        self.temporal_frames = temporal_frames
        self.base_patches = expected_num_patches // temporal_frames

    @contextlib.contextmanager
    def cache_context(self, name: str):
        """Dummy cache context for compatibility with pipeline."""
        yield

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, img_shapes=None, return_dict=False, **kwargs):
        """Forward pass using V1 Flash compiled transformer with RoPE as input."""
        batch_size = hidden_states.shape[0]

        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG Transformer V1 Flash input shapes:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"  timestep: {timestep.shape}")
            print(f"  img_rotary_emb: {self.img_rotary_emb.shape}")
            print(f"  txt_rotary_emb: {self.txt_rotary_emb.shape}")
            print(f"  temporal_frames: {self.temporal_frames}, base_patches: {self.base_patches}")
            self._debug_printed = True

        # Handle hidden_states padding
        actual_patches = hidden_states.shape[1]
        if actual_patches != self.expected_num_patches:
            if actual_patches < self.expected_num_patches:
                pad_size = self.expected_num_patches - actual_patches
                padding = torch.zeros(
                    (batch_size, pad_size, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            else:
                print(f"ERROR: hidden_states has {actual_patches} patches but model expects {self.expected_num_patches}")
                hidden_states = hidden_states[:, :self.expected_num_patches, :]

        # Handle encoder_hidden_states padding
        actual_seq_len = encoder_hidden_states.shape[1]
        if actual_seq_len != self.expected_seq_len:
            if actual_seq_len < self.expected_seq_len:
                pad_size = self.expected_seq_len - actual_seq_len
                padding = torch.zeros(
                    (batch_size, pad_size, encoder_hidden_states.shape[2]),
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
            else:
                print(f"WARNING: Truncating encoder_hidden_states from {actual_seq_len} to {self.expected_seq_len}")
                encoder_hidden_states = encoder_hidden_states[:, :self.expected_seq_len, :]

        # Convert timestep to float32
        timestep = timestep.to(torch.float32)

        # Run compiled transformer with RoPE as input
        import time
        _t_start = time.time()
        output = self.compiled_transformer(
            hidden_states,
            encoder_hidden_states,
            timestep,
            self.img_rotary_emb,
            self.txt_rotary_emb
        )

        # Profile
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._step_times = []
        self._step_count += 1
        self._step_times.append(time.time() - _t_start)
        if self._step_count % 10 == 0:
            avg_time = sum(self._step_times[-10:]) / 10
            print(f"  [Profile] Transformer V1 Flash (Neuron) step {self._step_count}: avg {avg_time:.3f}s/step")

        # Extract tensor from output
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Extract first frame as noise prediction (same as V2)
        output_tensor = output_tensor[:, :self.base_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


def load_transformer_v1_flash(compiled_models_dir: str, pipe, args):
    """
    Load V1 Flash compiled transformer model using parallel_model_load.

    V1 Flash models are compiled with parallel_model_trace and require:
    1. Model files in transformer_v1_flash/ directory
    2. rope_cache.pt - pre-computed RoPE tensors
    3. config.json - model configuration

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Pipeline with original transformer (for config)
        args: Command line arguments

    Returns:
        NeuronTransformerWrapperV1Flash wrapping the loaded model
    """
    import json

    v1_flash_path = f"{compiled_models_dir}/transformer_v1_flash"
    model_path = f"{v1_flash_path}/model"  # Model files are in subdirectory
    rope_cache_path = f"{v1_flash_path}/rope_cache.pt"
    config_path = f"{v1_flash_path}/config.json"

    # Validate files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"V1 Flash transformer not found at {model_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v1_flash.py"
        )
    if not os.path.exists(rope_cache_path):
        raise FileNotFoundError(
            f"V1 Flash RoPE cache not found at {rope_cache_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v1_flash.py"
        )

    # Load config
    print(f"  Loading V1 Flash config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    expected_num_patches = config["num_patches"]
    expected_seq_len = config["text_seq_len"]
    temporal_frames = config.get("frame", config.get("patch_multiplier", 3))
    base_patches = expected_num_patches // temporal_frames
    print(f"  V1 Flash config: patches={expected_num_patches}, seq_len={expected_seq_len}")
    print(f"  V1 Flash config: temporal_frames={temporal_frames}, base_patches={base_patches}")
    print(f"  NKI Flash Attention: {config.get('nki_flash_attention', False)}")

    # Load pre-computed RoPE tensors
    print(f"  Loading RoPE cache from {rope_cache_path}...")
    rope_cache = torch.load(rope_cache_path)
    img_rotary_emb = rope_cache["img_rotary_emb"].to(torch.bfloat16)
    txt_rotary_emb = rope_cache["txt_rotary_emb"].to(torch.bfloat16)
    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    # Load compiled model using parallel_model_load (from model subdirectory)
    print(f"  Loading V1 Flash model from {model_path}...")
    compiled_transformer = neuronx_distributed.trace.parallel_model_load(model_path)
    print("  V1 Flash model loaded!")

    # Create wrapper
    wrapper = NeuronTransformerWrapperV1Flash(
        original_transformer=pipe.transformer,
        compiled_transformer=compiled_transformer,
        img_rotary_emb=img_rotary_emb,
        txt_rotary_emb=txt_rotary_emb,
        expected_num_patches=expected_num_patches,
        expected_seq_len=expected_seq_len,
        temporal_frames=temporal_frames,
    )

    return wrapper


def load_transformer_v2_flash(compiled_models_dir: str, pipe, args):
    """
    Load V2 Flash compiled transformer model using NxDModel API.

    V2 Flash models combine ModelBuilder API with NKI Flash Attention:
    1. nxd_model.pt - the compiled model
    2. weights/ - sharded checkpoints
    3. rope_cache.pt - pre-computed RoPE tensors
    4. config.json - model configuration

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Pipeline with original transformer (for config)
        args: Command line arguments

    Returns:
        NeuronTransformerWrapperV2 wrapping the loaded model (reuses V2 wrapper)
    """
    import json

    if not NXD_MODEL_AVAILABLE:
        raise RuntimeError(
            "NxDModel is not available. Please ensure neuronx_distributed is installed correctly."
        )

    v2_flash_path = f"{compiled_models_dir}/transformer_v2_flash"
    nxd_model_path = f"{v2_flash_path}/nxd_model.pt"
    weights_path = f"{v2_flash_path}/weights"
    rope_cache_path = f"{v2_flash_path}/rope_cache.pt"
    config_path = f"{v2_flash_path}/config.json"

    # Validate all required files exist
    if not os.path.exists(nxd_model_path):
        raise FileNotFoundError(
            f"V2 Flash transformer model not found at {nxd_model_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2_flash.py"
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"V2 Flash transformer weights not found at {weights_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2_flash.py"
        )
    if not os.path.exists(rope_cache_path):
        raise FileNotFoundError(
            f"V2 Flash RoPE cache not found at {rope_cache_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_transformer_v2_flash.py"
        )

    # Load config
    print(f"  Loading V2 Flash config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    expected_num_patches = config["num_patches"]
    expected_seq_len = config["text_seq_len"]
    temporal_frames = config.get("frame", config.get("patch_multiplier", 3))
    base_patches = expected_num_patches // temporal_frames
    print(f"  V2 Flash config: patches={expected_num_patches}, seq_len={expected_seq_len}")
    print(f"  V2 Flash config: temporal_frames={temporal_frames}, base_patches={base_patches}")
    print(f"  NKI Flash Attention: {config.get('nki_flash_attention', False)}")

    # Load pre-computed RoPE tensors
    print(f"  Loading RoPE cache from {rope_cache_path}...")
    rope_cache = torch.load(rope_cache_path)
    img_rotary_emb = rope_cache["img_rotary_emb"].to(torch.bfloat16)
    txt_rotary_emb = rope_cache["txt_rotary_emb"].to(torch.bfloat16)
    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    # Load the compiled model using NxDModel.load()
    print(f"  Loading V2 Flash model from {nxd_model_path}...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded checkpoints
    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 8)
    print(f"  Loading sharded weights for TP={tp_degree}...")
    sharded_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_file(ckpt_path)
        sharded_checkpoints.append(ckpt)
        if rank == 0:
            print(f"    Rank 0 checkpoint keys: {len(ckpt)} tensors")

    # Initialize model with weights and move to Neuron
    print("  Setting weights...")
    nxd_model.set_weights(sharded_checkpoints)
    print("  Moving model to Neuron...")
    nxd_model.to_neuron()
    print("  V2 Flash model initialized on Neuron!")

    # Create wrapper (reuse V2 wrapper since interface is the same)
    wrapper = NeuronTransformerWrapperV2(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        img_rotary_emb=img_rotary_emb,
        txt_rotary_emb=txt_rotary_emb,
        expected_num_patches=expected_num_patches,
        expected_seq_len=expected_seq_len,
        temporal_frames=temporal_frames,
    )

    return wrapper


class NeuronTransformerWrapperV3CP(torch.nn.Module):
    """
    Wrapper for V3 CP (Context Parallel) compiled transformer.

    Key features:
    - Uses TP=4, CP=2 (world_size=8)
    - K/V are all-gathered across CP group before attention
    - Each CP rank processes part of the sequence
    - RoPE is sharded per CP rank
    """
    def __init__(self, original_transformer, nxd_model, img_rotary_emb, txt_rotary_emb,
                 expected_num_patches=1024, expected_seq_len=512, temporal_frames=3,
                 cp_degree=2):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        # Full RoPE (will be sharded at runtime per CP rank)
        self.img_rotary_emb_full = img_rotary_emb
        self.txt_rotary_emb_full = txt_rotary_emb

        self.expected_num_patches = expected_num_patches
        self.expected_seq_len = expected_seq_len
        self.temporal_frames = temporal_frames
        self.base_patches = expected_num_patches // temporal_frames
        self.cp_degree = cp_degree

        # Local dimensions (per CP rank)
        self.local_num_patches = expected_num_patches // cp_degree
        self.local_seq_len = expected_seq_len // cp_degree

    @contextlib.contextmanager
    def cache_context(self, name: str):
        """Dummy cache context for compatibility with pipeline."""
        yield

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, img_shapes=None, return_dict=False, **kwargs):
        """Forward pass with Context Parallel."""
        batch_size = hidden_states.shape[0]

        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG Transformer V3 CP input shapes:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"  timestep: {timestep.shape}")
            print(f"  img_rotary_emb_full: {self.img_rotary_emb_full.shape}")
            print(f"  txt_rotary_emb_full: {self.txt_rotary_emb_full.shape}")
            print(f"  CP degree: {self.cp_degree}")
            print(f"  Local patches: {self.local_num_patches}, Local seq_len: {self.local_seq_len}")
            # Print tensor statistics for debugging
            print(f"  hidden_states stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
            print(f"  encoder_hidden_states stats: min={encoder_hidden_states.min():.4f}, max={encoder_hidden_states.max():.4f}")
            self._debug_printed = True

        # For CP, the model expects LOCAL data (already sharded)
        # Since we're running inference, we pass full data and let the model handle it
        # The compiled model has the gather/scatter logic built in

        # Handle hidden_states padding to expected_num_patches
        actual_patches = hidden_states.shape[1]
        if actual_patches != self.expected_num_patches:
            if actual_patches < self.expected_num_patches:
                pad_size = self.expected_num_patches - actual_patches
                padding = torch.zeros(
                    (batch_size, pad_size, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            else:
                print(f"ERROR: hidden_states has {actual_patches} patches but model expects {self.expected_num_patches}")
                hidden_states = hidden_states[:, :self.expected_num_patches, :]

        # Handle encoder_hidden_states padding
        actual_seq_len = encoder_hidden_states.shape[1]
        if actual_seq_len != self.expected_seq_len:
            if actual_seq_len < self.expected_seq_len:
                pad_size = self.expected_seq_len - actual_seq_len
                padding = torch.zeros(
                    (batch_size, pad_size, encoder_hidden_states.shape[2]),
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
            else:
                print(f"WARNING: Truncating encoder_hidden_states from {actual_seq_len} to {self.expected_seq_len}")
                encoder_hidden_states = encoder_hidden_states[:, :self.expected_seq_len, :]

        # Convert timestep to float32
        timestep = timestep.to(torch.float32)

        # Run model
        # Note: For CP models compiled with ModelBuilder, the sharding is handled internally
        # We pass full data and full RoPE - the model handles the rest
        import time
        _t_start = time.time()
        output = self.nxd_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            self.img_rotary_emb_full,
            self.txt_rotary_emb_full
        )

        # Profile
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._step_times = []
        self._step_count += 1
        self._step_times.append(time.time() - _t_start)
        if self._step_count % 10 == 0:
            avg_time = sum(self._step_times[-10:]) / 10
            print(f"  [Profile] Transformer V3 CP (Neuron) step {self._step_count}: avg {avg_time:.3f}s/step")

        # Extract tensor from output
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Debug: Print output stats on first few calls
        if not hasattr(self, '_output_debug_count'):
            self._output_debug_count = 0
        if self._output_debug_count < 3:
            print(f"DEBUG V3 CP output (step {self._output_debug_count}):")
            print(f"  output shape: {output_tensor.shape}")
            print(f"  output stats: min={output_tensor.min():.4f}, max={output_tensor.max():.4f}, mean={output_tensor.mean():.4f}")
            # Check if output looks reasonable (not all zeros, no NaN/Inf)
            if torch.isnan(output_tensor).any():
                print("  WARNING: NaN values detected in output!")
            if torch.isinf(output_tensor).any():
                print("  WARNING: Inf values detected in output!")
            if (output_tensor == 0).all():
                print("  WARNING: Output is all zeros!")
            self._output_debug_count += 1

        # Extract first frame as noise prediction
        output_tensor = output_tensor[:, :self.base_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


def load_transformer_v3_cp(compiled_models_dir: str, pipe, args):
    """
    Load V3 CP compiled transformer with Context Parallel.

    V3 CP models use:
    - TP=4, CP=2 (world_size=8)
    - K/V all-gather across CP group
    - NKI Flash Attention

    Args:
        compiled_models_dir: Directory containing compiled model artifacts
        pipe: Pipeline with original transformer (for config)
        args: Command line arguments

    Returns:
        NeuronTransformerWrapperV3CP wrapping the loaded model
    """
    import json

    if not NXD_MODEL_AVAILABLE:
        raise RuntimeError(
            "NxDModel is not available. Please ensure neuronx_distributed is installed correctly."
        )

    v3_cp_path = f"{compiled_models_dir}/transformer_v3_cp"
    nxd_model_path = f"{v3_cp_path}/nxd_model.pt"
    weights_path = f"{v3_cp_path}/weights"
    rope_cache_path = f"{v3_cp_path}/rope_cache.pt"
    config_path = f"{v3_cp_path}/config.json"

    # Validate files exist
    if not os.path.exists(nxd_model_path):
        raise FileNotFoundError(
            f"V3 CP transformer model not found at {nxd_model_path}\n"
            "Please run: ./compile.sh v3_cp"
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"V3 CP transformer weights not found at {weights_path}\n"
            "Please run: ./compile.sh v3_cp"
        )
    if not os.path.exists(rope_cache_path):
        raise FileNotFoundError(
            f"V3 CP RoPE cache not found at {rope_cache_path}\n"
            "Please run: ./compile.sh v3_cp"
        )

    # Load config
    print(f"  Loading V3 CP config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    expected_num_patches = config["num_patches"]
    expected_seq_len = config["text_seq_len"]
    temporal_frames = config.get("frame", config.get("patch_multiplier", 3))
    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)
    cp_degree = config.get("cp_degree", 2)
    base_patches = expected_num_patches // temporal_frames

    print(f"  V3 CP config: patches={expected_num_patches}, seq_len={expected_seq_len}")
    print(f"  V3 CP config: temporal_frames={temporal_frames}, base_patches={base_patches}")
    print(f"  V3 CP config: TP={tp_degree}, world_size={world_size}, CP={cp_degree}")
    print(f"  Context Parallel: {config.get('context_parallel', False)}")
    print(f"  NKI Flash Attention: {config.get('nki_flash_attention', False)}")

    # Load pre-computed RoPE tensors (full, not sharded)
    print(f"  Loading RoPE cache from {rope_cache_path}...")
    rope_cache = torch.load(rope_cache_path)
    img_rotary_emb = rope_cache["img_rotary_emb"].to(torch.bfloat16)
    txt_rotary_emb = rope_cache["txt_rotary_emb"].to(torch.bfloat16)
    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    # Load the compiled model using NxDModel.load()
    print(f"  Loading V3 CP model from {nxd_model_path}...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded checkpoints
    # For Context Parallel: TP=4 but world_size=8
    # Each DP rank (CP rank) uses the same weights as its corresponding TP rank
    # So we need to duplicate: [tp0, tp1, tp2, tp3] -> [tp0, tp1, tp2, tp3, tp0, tp1, tp2, tp3]
    from safetensors.torch import load_file
    print(f"  Loading sharded weights for TP={tp_degree}, world_size={world_size}...")

    # First load the TP checkpoints
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)
        if rank == 0:
            print(f"    Rank 0 checkpoint keys: {len(ckpt)} tensors")

    # For CP, duplicate checkpoints for each DP rank
    # world_size = tp_degree * dp_degree (dp_degree = cp_degree)
    # IMPORTANT: Each world rank needs a unique global_rank value for SPMD scatter/gather
    sharded_checkpoints = []
    for dp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            # Clone the checkpoint so we can modify global_rank independently
            world_rank = dp_rank * tp_degree + tp_rank
            ckpt_copy = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}

            # Set the correct global_rank for this world rank
            # This is CRITICAL for SPMDRank to return the correct rank at runtime
            global_rank_key = 'transformer.global_rank.rank'
            if global_rank_key in ckpt_copy:
                ckpt_copy[global_rank_key] = torch.tensor([world_rank], dtype=torch.int32)
                if world_rank < 2 or world_rank >= world_size - 2:
                    print(f"    World rank {world_rank}: global_rank set to {world_rank}")

            sharded_checkpoints.append(ckpt_copy)

    print(f"  Total checkpoints: {len(sharded_checkpoints)} (TP={tp_degree} x CP={cp_degree})")
    print(f"  Each world rank has unique global_rank for SPMD execution")

    # Initialize model with weights and move to Neuron
    print("  Setting weights...")
    # Debug: Verify global_rank values in checkpoints
    for i in [0, 4]:  # Check first rank of each DP group
        if i < len(sharded_checkpoints):
            ckpt = sharded_checkpoints[i]
            gr_key = 'transformer.global_rank.rank'
            if gr_key in ckpt:
                print(f"    Checkpoint[{i}] global_rank = {ckpt[gr_key].item()}")
            else:
                print(f"    WARNING: Checkpoint[{i}] missing {gr_key}")
    nxd_model.set_weights(sharded_checkpoints)
    print("  Moving model to Neuron...")
    nxd_model.to_neuron()
    print("  V3 CP model initialized on Neuron!")

    # Create wrapper
    wrapper = NeuronTransformerWrapperV3CP(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        img_rotary_emb=img_rotary_emb,
        txt_rotary_emb=txt_rotary_emb,
        expected_num_patches=expected_num_patches,
        expected_seq_len=expected_seq_len,
        temporal_frames=temporal_frames,
        cp_degree=cp_degree,
    )

    return wrapper


def load_language_model_v3(compiled_models_dir: str):
    """
    Load V3 compiled language model using NxDModel.

    V3 language models use:
    - TP=4, world_size=8 (matching V3 CP transformer)
    - ModelBuilder API (NxDModel)

    Note: Unlike V3 CP transformer which splits sequence (Context Parallel),
    the language model processes the full sequence on all ranks.
    Checkpoints are simply duplicated for world_size=8.

    Returns:
        NxDModel wrapping the loaded language model
    """
    import json

    if not NXD_MODEL_AVAILABLE:
        raise RuntimeError(
            "NxDModel is not available. Please ensure neuronx_distributed is installed correctly."
        )

    v3_path = f"{compiled_models_dir}/language_model_v3"
    nxd_model_path = f"{v3_path}/nxd_model.pt"
    weights_path = f"{v3_path}/weights"
    config_path = f"{v3_path}/config.json"

    # Validate files exist
    if not os.path.exists(nxd_model_path):
        raise FileNotFoundError(
            f"V3 language model not found at {nxd_model_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_language_model_v3.py"
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"V3 language model weights not found at {weights_path}\n"
            "Please run: python neuron_qwen_image_edit/compile_language_model_v3.py"
        )

    # Load config
    print(f"  Loading V3 language model config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)
    max_seq_len = config.get("max_sequence_length", 1024)
    cp_degree = world_size // tp_degree  # 2

    print(f"  V3 language model config:")
    print(f"    TP={tp_degree}, world_size={world_size}")
    print(f"    max_sequence_length={max_seq_len}")
    print(f"    GQA: 28Q/4=7 heads/rank, 4KV/4=1 head/rank (perfect fit)")

    # Load the compiled model using NxDModel.load()
    print(f"  Loading V3 language model from {nxd_model_path}...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded checkpoints
    # For world_size=8 with TP=4: duplicate TP checkpoints for each CP rank
    from safetensors.torch import load_file
    print(f"  Loading sharded weights for TP={tp_degree}, world_size={world_size}...")

    # First load the TP checkpoints (only tp_degree files exist)
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)
        if rank == 0:
            print(f"    Rank 0 checkpoint keys: {len(ckpt)} tensors")

    # Duplicate for world_size=8
    # Unlike transformer CP which needs different global_rank values,
    # language model processes full sequence on all ranks (no CP scatter/gather)
    # So we simply duplicate the TP checkpoints
    sharded_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            # Clone the checkpoint
            ckpt_copy = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            sharded_checkpoints.append(ckpt_copy)

    print(f"  Total checkpoints: {len(sharded_checkpoints)} (TP={tp_degree} x CP={cp_degree})")

    # Initialize model with weights and move to Neuron
    print("  Setting weights...")
    nxd_model.set_weights(sharded_checkpoints)
    print("  Moving model to Neuron...")
    nxd_model.to_neuron()
    print("  V3 language model initialized on Neuron!")

    return nxd_model, config


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
        import time
        _t_enc_start = time.time()

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

            print(f"  [Profile] VAE encode (Neuron): {time.time() - _t_enc_start:.2f}s")
            return EncoderOutput(LatentDist(sample, mean))
        print(f"  [Profile] VAE encode (Neuron): {time.time() - _t_enc_start:.2f}s")
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
        import time
        _t_dec_start = time.time()

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

        print(f"  [Profile] VAE decode (Neuron): {time.time() - _t_dec_start:.2f}s")
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
    # CPU is the default for better accuracy, use --neuron_vision_encoder to use Neuron
    vision_encoder_tp_path = f"{compiled_models_dir}/vision_encoder_tp"
    use_vision_tp = args.vision_tp if hasattr(args, 'vision_tp') else False
    use_neuron_vision = getattr(args, 'neuron_vision_encoder', False)  # Default to CPU
    use_cpu_vision_encoder = not use_neuron_vision
    if use_cpu_vision_encoder:
        vision_mode = "CPU (default)"
    elif use_vision_tp or os.path.exists(vision_encoder_tp_path):
        vision_mode = "Neuron TP=8"
    else:
        vision_mode = "Neuron (float32)"

    print("\n" + "=" * 60)
    print("Loading Compiled Models for Trainium2")
    print("=" * 60)
    # Check language model mode
    # Priority: --use_v3_language_model > --neuron_language_model > --cpu_language_model (default)
    use_v3_language_model = getattr(args, 'use_v3_language_model', False)
    use_neuron_language_model = getattr(args, 'neuron_language_model', False)
    use_cpu_language_model = not (use_v3_language_model or use_neuron_language_model)

    if use_v3_language_model:
        language_mode = "Neuron V3 (TP=4, world_size=8)"
    elif use_neuron_language_model:
        language_mode = "Neuron (TP=8, KV replication)"
    else:
        language_mode = "CPU"

    print("Parallel configuration:")
    print("  - VAE: Single device (avoid collective conflict)")
    print("  - Transformer: TP=8")
    print(f"  - Vision Encoder: {vision_mode}")
    print(f"  - Language Model: {language_mode}")
    if use_cpu_language_model:
        print("\nNOTE: Language Model on CPU (safe fallback mode)")
        print("      Use --use_v3_language_model for V3 compiled model (recommended with --use_v3_cp)")
    elif use_v3_language_model:
        print("\nNOTE: Language Model uses V3 (ModelBuilder API)")
        print("      TP=4, world_size=8 - compatible with V3 CP transformer")
    else:
        print("\nNOTE: Language Model uses TP=8 with KV head replication")
        print("      (Q heads padded 28->32, KV heads replicated 4->8)")

    # ========================================
    # 1. Load Transformer FIRST (TP=8)
    # ========================================
    # IMPORTANT: Must load the largest TP model first to initialize
    # the communicator with the correct world size
    use_v2 = getattr(args, 'use_v2', False)
    use_v1_flash = getattr(args, 'use_v1_flash', False)
    use_v2_flash = getattr(args, 'use_v2_flash', False)
    use_v3_cp = getattr(args, 'use_v3_cp', False)
    v2_available = os.path.exists(f"{compiled_models_dir}/transformer_v2/nxd_model.pt")
    v1_flash_available = os.path.exists(f"{compiled_models_dir}/transformer_v1_flash")
    v2_flash_available = os.path.exists(f"{compiled_models_dir}/transformer_v2_flash/nxd_model.pt")
    v3_cp_available = os.path.exists(f"{compiled_models_dir}/transformer_v3_cp/nxd_model.pt")

    if use_v3_cp:
        print("\n[1/3] Loading Transformer V3 CP (Context Parallel + NKI Flash Attention, TP=4, CP=2)...")
        if not v3_cp_available:
            raise FileNotFoundError(
                f"V3 CP transformer not found. Please run: ./compile.sh v3_cp"
            )

        # Store reference to original for wrapper
        original_transformer = pipe.transformer

        # Load V3 CP model and assign to pipe
        pipe.transformer = load_transformer_v3_cp(compiled_models_dir, pipe, args)

        # Delete original transformer to free memory
        del original_transformer
        import gc
        gc.collect()
        print("  Transformer V3 CP loaded!")
        print("  Original transformer deleted to free memory.")
    elif use_v2_flash:
        print("\n[1/3] Loading Transformer V2 Flash (ModelBuilder + NKI Flash Attention, TP=8)...")
        if not v2_flash_available:
            raise FileNotFoundError(
                f"Transformer V2 Flash not found at {compiled_models_dir}/transformer_v2_flash\n"
                "Please run: python neuron_qwen_image_edit/compile_transformer_v2_flash.py"
            )

        # Store reference to original for wrapper
        original_transformer = pipe.transformer

        # Load V2 Flash model
        pipe.transformer = load_transformer_v2_flash(compiled_models_dir, pipe, args)

        # Delete original transformer to free ~40GB memory
        del original_transformer
        gc.collect()
        print("  Transformer V2 Flash loaded!")
        print("  Original transformer deleted to free memory.")
    elif use_v1_flash:
        print("\n[1/3] Loading Transformer V1 Flash (parallel_model_trace + NKI Flash Attention, TP=8)...")
        if not v1_flash_available:
            raise FileNotFoundError(
                f"Transformer V1 Flash not found at {compiled_models_dir}/transformer_v1_flash\n"
                "Please run: python neuron_qwen_image_edit/compile_transformer_v1_flash.py"
            )

        # Store reference to original for wrapper
        original_transformer = pipe.transformer

        # Load V1 Flash model
        pipe.transformer = load_transformer_v1_flash(compiled_models_dir, pipe, args)

        # Delete original transformer to free ~40GB memory
        del original_transformer
        gc.collect()
        print("  Transformer V1 Flash loaded!")
        print("  Original transformer deleted to free memory.")
    elif use_v2:
        print("\n[1/3] Loading Transformer V2 (ModelBuilder API, TP=8)...")
        if not v2_available:
            raise FileNotFoundError(
                f"Transformer V2 not found at {compiled_models_dir}/transformer_v2\n"
                "Please run: python neuron_qwen_image_edit/compile_transformer_v2.py"
            )

        # Store reference to original for wrapper
        original_transformer = pipe.transformer

        # Load V2 model
        pipe.transformer = load_transformer_v2(compiled_models_dir, pipe, args)

        # Delete original transformer to free ~40GB memory
        del original_transformer
        gc.collect()
        print("  Transformer V2 loaded!")
        print("  Original transformer deleted to free memory.")
    else:
        print("\n[1/3] Loading Transformer V1 (parallel_model_trace API, TP=8)...")

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

        # Calculate expected shapes based on image dimensions
        latent_h = args.height // 8
        latent_w = args.width // 8
        patch_h = latent_h // 2
        patch_w = latent_w // 2
        base_num_patches = patch_h * patch_w  # e.g., 64*64=4096 for 1024x1024

        # For IMAGE EDITING, patches are doubled (source + noise latents concatenated)
        # This is handled by using temporal_frames = patch_multiplier
        # - patch_multiplier=1 (generation): temporal_frames=1, patches = 1 * 32 * 32 = 1024
        # - patch_multiplier=2 (editing): temporal_frames=2, patches = 2 * 32 * 32 = 2048
        temporal_frames = args.patch_multiplier
        expected_num_patches = temporal_frames * base_num_patches
        print(f"  Expected num_patches: {expected_num_patches} (temporal_frames={temporal_frames}, base={base_num_patches})")

        # img_shapes for the wrapper
        # Note: batch_size=1, CFG runs transformer twice sequentially (not batch_size=2)
        img_shapes = [(temporal_frames, patch_h, patch_w)]

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
        print(f"  Transformer V1 loaded (TP=8)! Expected patches={expected_num_patches}, seq_len={args.max_sequence_length}")
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
        # Load single-device vision encoder (always float32)
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
        print(f"  Vision encoder loaded (single device, float32)!")

    # Load Language Model
    compiled_language_model = None
    compiled_language_model_v3 = None
    cpu_language_model = None
    language_model_config = None

    if use_v3_language_model:
        # V3 Language Model mode - uses ModelBuilder API with TP=4, world_size=8
        # Compatible with V3 CP transformer
        print("  Loading V3 Language Model (TP=4, world_size=8)...")
        compiled_language_model_v3, language_model_config = load_language_model_v3(compiled_models_dir)
        print("  V3 Language model loaded!")
    elif use_cpu_language_model:
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
        compiled_language_model_v3=compiled_language_model_v3,
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
    elif use_v3_language_model:
        # V3 model loaded, can delete original
        del original_text_encoder
        gc.collect()
        print("  Text encoder wrapper created!")
        print("  Original text encoder deleted to free memory.")
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
    if use_v3_cp:
        transformer_api = "V3 CP (Context Parallel + NKI, TP=4, CP=2)"
        tp_info = "TP=4, CP=2"
    elif use_v2_flash:
        transformer_api = "V2 Flash (ModelBuilder + NKI)"
        tp_info = "TP=8"
    elif use_v1_flash:
        transformer_api = "V1 Flash (parallel_model_trace + NKI)"
        tp_info = "TP=8"
    elif use_v2:
        transformer_api = "V2 (ModelBuilder)"
        tp_info = "TP=8"
    else:
        transformer_api = "V1 (parallel_model_trace)"
        tp_info = "TP=8"
    print("\n" + "=" * 60)
    print("All Models Loaded!")
    print("=" * 60)
    print(f"  - Transformer: Neuron ({tp_info}, {transformer_api})")
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
    print(f"  Compiled dimensions: {args.height}x{args.width}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  CFG scale: {args.true_cfg_scale}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    dtype = torch.bfloat16

    # CRITICAL FIX: Override VAE_IMAGE_SIZE before loading pipeline
    # The pipeline uses VAE_IMAGE_SIZE (default 1024*1024) to resize source images.
    # This creates more patches than our compiled transformer expects.
    # We need to match our compiled dimensions.
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_pipeline_module
    compiled_vae_pixels = args.height * args.width  # e.g., 512*512
    original_vae_size = getattr(qwen_pipeline_module, 'VAE_IMAGE_SIZE', 1024*1024)
    qwen_pipeline_module.VAE_IMAGE_SIZE = compiled_vae_pixels
    print(f"\nOverriding VAE_IMAGE_SIZE: {original_vae_size} -> {compiled_vae_pixels}")
    print(f"  (This ensures source images produce {args.height//8//2}x{args.width//8//2} patches)")

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
        img = img.resize((args.width, args.height))
        source_images.append(img)
    print(f"All images resized to: {args.width}x{args.height} (compiled dimensions)")

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

    # CFG is controlled by true_cfg_scale (default 4.0 in pipeline)
    # CFG runs transformer twice sequentially, NOT with batch_size=2
    true_cfg_scale = args.true_cfg_scale

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
            height=args.height,  # Use compiled dimensions
            width=args.width,
            true_cfg_scale=true_cfg_scale,
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
        height=args.height,  # Use compiled dimensions
        width=args.width,
        true_cfg_scale=true_cfg_scale,
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

    # Image dimensions (must match compiled model)
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height (must match compiled model)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width (must match compiled model)")
    parser.add_argument("--patch_multiplier", type=int, default=2,
                        help="Patch multiplier (2 for image editing, 1 for generation)")

    # Text encoder settings - MUST match compilation settings
    parser.add_argument("--image_size", type=int, default=448,
                        help="Vision encoder image size (must match compiled model)")
    parser.add_argument("--max_sequence_length", type=int, default=1024,
                        help="Max text sequence length (must match compiled model)")
    parser.add_argument("--vision_tp", action="store_true",
                        help="Use TP-compiled vision encoder (from vision_encoder_tp/). "
                             "Default is to auto-detect based on available compiled models.")

    # Language model mode
    parser.add_argument("--cpu_language_model", action="store_true", default=True,
                        help="Run Language Model on CPU (default). "
                             "Safe fallback mode that avoids any TP compatibility issues.")
    parser.add_argument("--neuron_language_model", action="store_true",
                        help="Use Neuron-compiled Language Model with TP=8 (KV head replication mode). "
                             "Requires: python compile_text_encoder.py --language_only --language_tp_degree 8")
    parser.add_argument("--use_v3_language_model", action=argparse.BooleanOptionalAction, default=True,
                        help="Use V3 Language Model compiled with ModelBuilder API (TP=4, world_size=8). "
                             "Default: True. Use --no-use_v3_language_model to disable. "
                             "Requires: python neuron_qwen_image_edit/compile_language_model_v3.py")

    # Vision encoder mode
    parser.add_argument("--cpu_vision_encoder", action="store_true",
                        help="Run Vision Encoder on CPU (default behavior)")
    parser.add_argument("--neuron_vision_encoder", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Neuron-compiled Vision Encoder (float32). "
                             "CPU is used by default for better accuracy.")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Number of denoising steps (default: 40)")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (default: 4.0). "
                             "CFG runs transformer twice sequentially (not batch_size=2).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")

    # Model settings
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--vae_tile_size", type=int, default=512,
                        help="VAE tile size (must match compiled VAE size). "
                             "For larger images, tiled VAE will process in this tile size.")
    parser.add_argument("--use_v2", action="store_true",
                        help="Use V2 transformer compiled with ModelBuilder API. "
                             "V2 passes RoPE as input tensors (like Flux). "
                             "Requires: python neuron_qwen_image_edit/compile_transformer_v2.py")
    parser.add_argument("--use_v1_flash", action="store_true",
                        help="Use V1 Flash transformer with NKI Flash Attention. "
                             "Combines V1's parallel_model_trace (supports NKI) with V2's RoPE handling. "
                             "Requires: python neuron_qwen_image_edit/compile_transformer_v1_flash.py")
    parser.add_argument("--use_v2_flash", action="store_true",
                        help="Use V2 Flash transformer with ModelBuilder + NKI Flash Attention. "
                             "Combines ModelBuilder's XLA optimization with NKI's hardware attention. "
                             "Requires: python neuron_qwen_image_edit/compile_transformer_v2_flash.py")
    parser.add_argument("--use_v3_cp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use V3 CP transformer with Context Parallel + NKI Flash Attention. "
                             "Default: True. Use --no-use_v3_cp to disable. "
                             "Requires: ./compile.sh v3_cp")

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
