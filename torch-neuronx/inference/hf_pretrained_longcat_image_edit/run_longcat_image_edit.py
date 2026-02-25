"""
LongCat-Image-Edit Inference Script for AWS Trainium2

Runs the LongCat-Image-Edit model ENTIRELY on Neuron devices.
All components (Text Encoder, FLUX Transformer, VAE) run on Trainium2.

Components:
- Text Encoder (Qwen2.5-VL): Vision encoder + Language model (TP=4)
- Transformer: LongCatImageTransformer2DModel (FLUX-style, TP=4, CP=2)
- VAE: 2D AutoencoderKL (single device)

Usage:
    # Single image editing:
    NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
        --image input.jpg --prompt "change the sky to sunset"

    # With warmup:
    NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
        --image input.jpg --prompt "make it look like a painting" --warmup
"""

import os

# ============================================================================
# CRITICAL: Set Neuron environment variables BEFORE any other imports
# ============================================================================
TP_DEGREE = 8  # For V3 CP: world_size=8 (TP=4, CP=2)

os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

print(f"Neuron runtime configured: world_size={TP_DEGREE}, LNC=2")

import argparse
import contextlib
import json
import random
import time

import numpy as np
import torch
import torch_neuronx
import neuronx_distributed
from PIL import Image

from diffusers import LongCatImageEditPipeline
from diffusers.utils import load_image

from neuron_longcat_image_edit.neuron_commons import NeuronTextEncoderWrapper

# Import NxDModel for V3 API loading
try:
    from neuronx_distributed.trace.nxd_model.nxd_model import NxDModel
    NXD_MODEL_AVAILABLE = True
except ImportError:
    NXD_MODEL_AVAILABLE = False
    print("WARNING: NxDModel not available.")

# Constants
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set to: {seed}")


class NeuronTransformerWrapperV3CP(torch.nn.Module):
    """
    Wrapper for V3 CP compiled LongCat FLUX transformer on Trainium2.

    Handles:
    - Padding hidden_states to expected_img_seq
    - Padding encoder_hidden_states to expected_txt_seq
    - Passing pre-computed RoPE (cos, sin)
    - Extracting target image patches from output
    """
    def __init__(self, original_transformer, nxd_model,
                 img_rotary_cos, img_rotary_sin,
                 txt_rotary_cos, txt_rotary_sin,
                 expected_img_patches=2048, expected_txt_seq=512,
                 target_patches=1024, batch_size=1):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        self.img_rotary_cos = img_rotary_cos
        self.img_rotary_sin = img_rotary_sin
        self.txt_rotary_cos = txt_rotary_cos
        self.txt_rotary_sin = txt_rotary_sin

        self.expected_img_patches = expected_img_patches
        self.expected_txt_seq = expected_txt_seq
        self.target_patches = target_patches
        self.compiled_batch_size = batch_size

    @contextlib.contextmanager
    def cache_context(self, name: str):
        yield

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, return_dict=False, **kwargs):
        """
        Forward pass using compiled V3 CP transformer.

        hidden_states: [B, img_patches, 64] -- packed latents for target+source
        encoder_hidden_states: [B, txt_seq, 3584] -- text embeddings
        timestep: [B] -- denoising timestep
        """
        batch_size = hidden_states.shape[0]

        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG V3 CP Transformer:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"  expected: img={self.expected_img_patches}, txt={self.expected_txt_seq}")
            print(f"  target_patches: {self.target_patches}")
            self._debug_printed = True

        # Pad hidden_states (image patches)
        actual_img = hidden_states.shape[1]
        if actual_img != self.expected_img_patches:
            if actual_img < self.expected_img_patches:
                pad = torch.zeros(
                    (batch_size, self.expected_img_patches - actual_img, hidden_states.shape[2]),
                    dtype=hidden_states.dtype, device=hidden_states.device)
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            else:
                hidden_states = hidden_states[:, :self.expected_img_patches, :]

        # Pad encoder_hidden_states (text)
        actual_txt = encoder_hidden_states.shape[1]
        if actual_txt != self.expected_txt_seq:
            if actual_txt < self.expected_txt_seq:
                pad = torch.zeros(
                    (batch_size, self.expected_txt_seq - actual_txt, encoder_hidden_states.shape[2]),
                    dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
                encoder_hidden_states = torch.cat([encoder_hidden_states, pad], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states[:, :self.expected_txt_seq, :]

        # Batch padding
        if batch_size < self.compiled_batch_size:
            pad_batch = self.compiled_batch_size - batch_size
            hidden_states = torch.cat([
                hidden_states,
                torch.zeros((pad_batch,) + hidden_states.shape[1:],
                           dtype=hidden_states.dtype, device=hidden_states.device)
            ], dim=0)
            encoder_hidden_states = torch.cat([
                encoder_hidden_states,
                torch.zeros((pad_batch,) + encoder_hidden_states.shape[1:],
                           dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            ], dim=0)
            timestep = torch.cat([
                timestep,
                torch.zeros(pad_batch, dtype=timestep.dtype, device=timestep.device)
            ], dim=0)

        timestep = timestep.to(torch.float32)

        # Run V3 CP model
        output = self.nxd_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            self.img_rotary_cos,
            self.img_rotary_sin,
            self.txt_rotary_cos,
            self.txt_rotary_sin,
        )

        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Remove batch padding
        if batch_size < self.compiled_batch_size:
            output_tensor = output_tensor[:batch_size]

        # Extract target image patches (first target_patches from output)
        output_tensor = output_tensor[:, :self.target_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


class _AttrDict(dict):
    """Dict that also supports attribute access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class NeuronVAEWrapper:
    """
    Wrapper for compiled 2D AutoencoderKL.

    Handles tiled encode/decode for images larger than compiled tile size.
    """
    def __init__(self, compiled_encoder, compiled_decoder, vae_config,
                 scaling_factor=0.3611, shift_factor=0.1159,
                 original_vae_config=None):
        self.compiled_encoder = compiled_encoder
        self.compiled_decoder = compiled_decoder
        # Use original VAE config if provided, otherwise wrap dict for attribute access
        if original_vae_config is not None:
            self.config = original_vae_config
        else:
            self.config = _AttrDict(vae_config)
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.tile_h = vae_config.get("height", 512)
        self.tile_w = vae_config.get("width", 512)
        self.latent_channels = vae_config.get("latent_channels", 16)
        self.dtype = torch.bfloat16

    def encode(self, x):
        """Encode image to latent space with tiled processing."""
        B, C, H, W = x.shape

        if H <= self.tile_h and W <= self.tile_w:
            moments = self.compiled_encoder(x)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            latents = mean  # Use mean for deterministic encoding
        else:
            latents = self._tiled_encode(x)

        # Scale latents
        latents = (latents - self.shift_factor) * self.scaling_factor
        # Return object compatible with retrieve_latents (only .latents, not .latent_dist)
        return _AttrDict(latents=latents)

    def decode(self, z, return_dict=False, **kwargs):
        """Decode latents to image with tiled processing."""
        # Unscale latents
        z = z / self.scaling_factor + self.shift_factor

        latent_h = z.shape[2]
        latent_w = z.shape[3]
        tile_latent_h = self.tile_h // 8
        tile_latent_w = self.tile_w // 8

        if latent_h <= tile_latent_h and latent_w <= tile_latent_w:
            result = self.compiled_decoder(z)
        else:
            result = self._tiled_decode(z)

        if return_dict:
            return _AttrDict(sample=result)
        return (result,)

    def _tiled_encode(self, x):
        """Tiled encoding for large images (no overlap, exact grid)."""
        B, C, H, W = x.shape
        tile_h, tile_w = self.tile_h, self.tile_w

        latent_tiles = []
        for y in range(0, H, tile_h):
            row_tiles = []
            for x_start in range(0, W, tile_w):
                y_end = min(y + tile_h, H)
                x_end = min(x_start + tile_w, W)
                tile = x[:, :, y:y_end, x_start:x_end]

                # Pad to tile size if needed
                actual_h, actual_w = tile.shape[2], tile.shape[3]
                if actual_h < tile_h or actual_w < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :actual_h, :actual_w] = tile
                    tile = padded

                moments = self.compiled_encoder(tile)
                mean, logvar = torch.chunk(moments, 2, dim=1)
                # Only keep the portion corresponding to actual input (not padding)
                row_tiles.append(mean[:, :, :(actual_h // 8), :(actual_w // 8)])
            latent_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in latent_tiles]
        return torch.cat(rows, dim=2)

    def _tiled_decode(self, z):
        """Tiled decoding for large latents (no overlap, exact grid)."""
        B, C, H, W = z.shape
        tile_h = self.tile_h // 8
        tile_w = self.tile_w // 8

        pixel_tiles = []
        for y in range(0, H, tile_h):
            row_tiles = []
            for x_start in range(0, W, tile_w):
                y_end = min(y + tile_h, H)
                x_end = min(x_start + tile_w, W)
                tile = z[:, :, y:y_end, x_start:x_end]

                actual_h, actual_w = tile.shape[2], tile.shape[3]
                if actual_h < tile_h or actual_w < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :actual_h, :actual_w] = tile
                    tile = padded

                decoded = self.compiled_decoder(tile)
                pixel_h = actual_h * 8
                pixel_w = actual_w * 8
                row_tiles.append(decoded[:, :, :pixel_h, :pixel_w])
            pixel_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in pixel_tiles]
        return torch.cat(rows, dim=2)


def load_transformer_v3_cp(compiled_models_dir, pipe, args):
    """Load V3 CP compiled transformer model."""
    v3_path = f"{compiled_models_dir}/transformer_v3_cp"
    nxd_model_path = f"{v3_path}/nxd_model.pt"
    weights_path = f"{v3_path}/weights"
    rope_cache_path = f"{v3_path}/rope_cache.pt"
    config_path = f"{v3_path}/config.json"

    for p, name in [(nxd_model_path, "model"), (weights_path, "weights"),
                     (rope_cache_path, "RoPE cache"), (config_path, "config")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"V3 CP {name} not found at {p}")

    with open(config_path, "r") as f:
        config = json.load(f)

    expected_img_patches = config["num_img_patches_padded"]
    expected_txt_seq = config["text_seq_len"]
    target_patches = config["num_img_patches"] // 2  # Only target image patches
    compiled_batch_size = config.get("batch_size", 1)

    print(f"  V3 CP config: img_patches={expected_img_patches}, txt_seq={expected_txt_seq}")
    print(f"  Target patches: {target_patches}, batch_size={compiled_batch_size}")

    # Load RoPE
    rope_cache = torch.load(rope_cache_path)
    img_rotary_cos = rope_cache["img_rotary_cos"].to(torch.bfloat16)
    img_rotary_sin = rope_cache["img_rotary_sin"].to(torch.bfloat16)
    txt_rotary_cos = rope_cache["txt_rotary_cos"].to(torch.bfloat16)
    txt_rotary_sin = rope_cache["txt_rotary_sin"].to(torch.bfloat16)

    # Load NxDModel
    print(f"  Loading V3 CP model...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded weights - need world_size checkpoints (TP*DP replicas)
    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)

    # Duplicate for DP replicas: world_size=8, tp=4 means 2 DP groups
    # Ranks [0,1,2,3] and [4,5,6,7] have identical weights
    sharded_checkpoints = []
    dp_size = world_size // tp_degree
    for dp_rank in range(dp_size):
        for tp_rank in range(tp_degree):
            sharded_checkpoints.append(tp_checkpoints[tp_rank])
    print(f"  Providing {len(sharded_checkpoints)} checkpoints for world_size={world_size}")

    nxd_model.set_weights(sharded_checkpoints)
    nxd_model.to_neuron()
    print("  V3 CP model initialized on Neuron!")

    wrapper = NeuronTransformerWrapperV3CP(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        img_rotary_cos=img_rotary_cos,
        img_rotary_sin=img_rotary_sin,
        txt_rotary_cos=txt_rotary_cos,
        txt_rotary_sin=txt_rotary_sin,
        expected_img_patches=expected_img_patches,
        expected_txt_seq=expected_txt_seq,
        target_patches=target_patches,
        batch_size=compiled_batch_size,
    )
    return wrapper


def load_text_encoder_v3(compiled_models_dir, pipe, args):
    """Load V3 compiled text encoder (vision encoder + language model)."""
    # Load vision encoder
    ve_path = f"{compiled_models_dir}/vision_encoder_v3"
    ve_model_path = f"{ve_path}/nxd_model.pt"

    compiled_ve = None
    if os.path.exists(ve_model_path):
        from safetensors.torch import load_file
        with open(f"{ve_path}/config.json") as f:
            ve_config = json.load(f)

        print("  Loading V3 vision encoder...")
        ve_nxd = NxDModel.load(ve_model_path)
        tp_degree = ve_config.get("tp_degree", 4)
        world_size = ve_config.get("world_size", 8)
        ve_tp_checkpoints = []
        for rank in range(tp_degree):
            ckpt = load_file(f"{ve_path}/weights/tp{rank}_sharded_checkpoint.safetensors")
            ve_tp_checkpoints.append(ckpt)
        # Duplicate for DP replicas
        ve_checkpoints = []
        dp_size = world_size // tp_degree
        for dp_rank in range(dp_size):
            for tp_rank in range(tp_degree):
                ve_checkpoints.append(ve_tp_checkpoints[tp_rank])
        ve_nxd.set_weights(ve_checkpoints)
        ve_nxd.to_neuron()
        compiled_ve = ve_nxd
        print("  V3 vision encoder loaded!")
    else:
        print("  WARNING: V3 vision encoder not found, will use CPU")

    # Load language model
    lm_path = f"{compiled_models_dir}/language_model_v3"
    lm_model_path = f"{lm_path}/nxd_model.pt"

    compiled_lm = None
    cpu_lm = None

    if os.path.exists(lm_model_path):
        from safetensors.torch import load_file
        with open(f"{lm_path}/config.json") as f:
            lm_config = json.load(f)

        print("  Loading V3 language model...")
        lm_nxd = NxDModel.load(lm_model_path)
        tp_degree = lm_config.get("tp_degree", 4)
        world_size = lm_config.get("world_size", 8)
        lm_tp_checkpoints = []
        for rank in range(tp_degree):
            ckpt = load_file(f"{lm_path}/weights/tp{rank}_sharded_checkpoint.safetensors")
            lm_tp_checkpoints.append(ckpt)
        # Duplicate for DP replicas
        lm_checkpoints = []
        dp_size = world_size // tp_degree
        for dp_rank in range(dp_size):
            for tp_rank in range(tp_degree):
                lm_checkpoints.append(lm_tp_checkpoints[tp_rank])
        lm_nxd.set_weights(lm_checkpoints)
        lm_nxd.to_neuron()
        compiled_lm = lm_nxd
        max_seq_len = lm_config.get("max_sequence_length", 512)
        lm_batch_size = lm_config.get("batch_size", 1)
        print("  V3 language model loaded!")
    else:
        print("  V3 language model not found, using CPU fallback")
        cpu_lm = pipe.text_encoder.model.language_model
        max_seq_len = 512
        lm_batch_size = 1

    # Create wrapper
    wrapper = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder_v3=compiled_ve,
        compiled_language_model_v3=compiled_lm,
        cpu_language_model=cpu_lm,
        image_size=args.image_size,
        max_seq_len=max_seq_len,
        language_model_batch_size=lm_batch_size,
    )
    return wrapper


def load_vae(compiled_models_dir, pipe):
    """Load compiled VAE."""
    encoder_path = f"{compiled_models_dir}/vae_encoder/model.pt"
    decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    config_path = f"{compiled_models_dir}/vae_config.json"

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("  WARNING: Compiled VAE not found, using CPU VAE")
        return pipe.vae

    with open(config_path) as f:
        vae_config = json.load(f)

    print(f"  Loading compiled VAE (tile: {vae_config['height']}x{vae_config['width']})")
    compiled_encoder = torch.jit.load(encoder_path)
    compiled_decoder = torch.jit.load(decoder_path)

    wrapper = NeuronVAEWrapper(
        compiled_encoder=compiled_encoder,
        compiled_decoder=compiled_decoder,
        vae_config=vae_config,
        scaling_factor=vae_config.get("scaling_factor", 0.3611),
        shift_factor=vae_config.get("shift_factor", 0.1159),
        original_vae_config=pipe.vae.config,
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="LongCat-Image-Edit Inference on Trainium2")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Edit instruction")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt")
    parser.add_argument("--output", type=str, default="output_edited.png", help="Output path")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--image_size", type=int, default=448, help="Vision encoder image size")
    parser.add_argument("--warmup", action="store_true", help="Run warmup inference")
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load pipeline
    print("\n[Step 1/4] Loading LongCat pipeline...")
    t0 = time.perf_counter()
    load_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if HUGGINGFACE_CACHE_DIR:
        load_kwargs["cache_dir"] = HUGGINGFACE_CACHE_DIR
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)
    print(f"  Pipeline loaded in {time.perf_counter() - t0:.1f}s")

    # Load compiled components
    print("\n[Step 2/4] Loading compiled Neuron models...")

    # Transformer
    print("Loading V3 CP transformer...")
    neuron_transformer = load_transformer_v3_cp(args.compiled_models_dir, pipe, args)

    # Text encoder
    print("Loading text encoder...")
    neuron_text_encoder = load_text_encoder_v3(args.compiled_models_dir, pipe, args)

    # VAE
    print("Loading VAE...")
    neuron_vae = load_vae(args.compiled_models_dir, pipe)

    # Replace pipeline components
    pipe.transformer = neuron_transformer
    pipe.text_encoder = neuron_text_encoder
    if not isinstance(neuron_vae, NeuronVAEWrapper):
        pass  # Keep original VAE if not compiled
    else:
        pipe.vae = neuron_vae

    # Monkey-patch device properties to return CPU (our Neuron wrappers handle device internally)
    type(pipe)._execution_device = property(lambda self: torch.device("cpu"))
    type(pipe).device = property(lambda self: torch.device("cpu"))

    # Delete original weights to save memory
    import gc
    gc.collect()

    # Load image
    print("\n[Step 3/4] Loading input image...")
    source_image = Image.open(args.image).convert("RGB")
    print(f"  Input image: {source_image.size}")

    # Run inference
    print(f"\n[Step 4/4] Running inference ({args.num_inference_steps} steps)...")

    if args.warmup:
        print("  Warmup run...")
        with torch.inference_mode():
            _ = pipe(
                image=source_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.manual_seed(args.seed),
            )
        print("  Warmup complete!")

    # Timed run
    set_seed(args.seed)
    t_start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=source_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.manual_seed(args.seed),
        )
    t_end = time.perf_counter()

    # Save output
    output_image = result.images[0]
    output_image.save(args.output)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Output saved to: {os.path.abspath(args.output)}")
    print(f"  Output size: {output_image.size}")
    print(f"  Total time: {t_end - t_start:.2f}s")
    print(f"  Steps/sec: {args.num_inference_steps / (t_end - t_start):.2f}")


if __name__ == "__main__":
    main()
