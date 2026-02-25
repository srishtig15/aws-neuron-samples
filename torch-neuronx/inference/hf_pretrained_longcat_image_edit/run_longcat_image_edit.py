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
# TP_DEGREE controls NxD world size. Use 4 for TP-only, 8 for TP+CP.
TP_DEGREE = int(os.environ.get("LONGCAT_WORLD_SIZE", "4"))

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

# Patch xm.mark_step() to no-op: the diffusers pipeline calls it inside the
# denoising loop, which attempts to synchronize ALL 64 NeuronCores on the
# machine. Since we only use a subset (e.g. 4 or 8), this hangs.
# The NxDModel handles its own synchronization internally.
try:
    import torch_xla.core.xla_model as xm
    xm.mark_step = lambda *args, **kwargs: None
except ImportError:
    pass

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
    - Computing RoPE at runtime from pipeline-provided position IDs
    - Padding hidden_states to expected_img_seq
    - Padding encoder_hidden_states to expected_txt_seq
    - Extracting target image patches from output
    """
    def __init__(self, original_transformer, nxd_model,
                 pos_embed, patch_h, patch_w,
                 expected_img_patches=8192, expected_txt_seq=512,
                 target_patches=4096, batch_size=1):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        # Keep pos_embed for runtime RoPE computation from pipeline position IDs
        self.pos_embed = pos_embed
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.expected_img_patches = expected_img_patches
        self.expected_txt_seq = expected_txt_seq
        self.target_patches = target_patches
        self.compiled_batch_size = batch_size

        # Cache RoPE keyed by (txt_len, img_len)
        self._rope_cache = {}

    @contextlib.contextmanager
    def cache_context(self, name: str):
        yield

    def _compute_rope_from_ids(self, txt_ids, img_ids):
        """Compute RoPE from pipeline-provided position IDs."""
        actual_txt = txt_ids.shape[0]
        actual_img = img_ids.shape[0]
        cache_key = (actual_txt, actual_img)

        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        # Pad txt_ids to expected_txt_seq
        if actual_txt < self.expected_txt_seq:
            pad_len = self.expected_txt_seq - actual_txt
            pad_ids = torch.zeros(pad_len, 3, dtype=txt_ids.dtype, device=txt_ids.device)
            last_row = txt_ids[-1, 1].item() if actual_txt > 0 else 0
            for i in range(pad_len):
                pad_ids[i, 0] = 0  # modality = text
                pad_ids[i, 1] = last_row + 1 + i
                pad_ids[i, 2] = last_row + 1 + i
            txt_ids_padded = torch.cat([txt_ids, pad_ids], dim=0)
        else:
            txt_ids_padded = txt_ids[:self.expected_txt_seq]

        # Pad img_ids to expected_img_patches
        if actual_img < self.expected_img_patches:
            pad_n = self.expected_img_patches - actual_img
            img_ids_padded = torch.cat(
                [img_ids, img_ids[-1:].expand(pad_n, -1)], dim=0)
        else:
            img_ids_padded = img_ids[:self.expected_img_patches]

        with torch.no_grad():
            txt_cos, txt_sin = self.pos_embed(txt_ids_padded)
            img_cos, img_sin = self.pos_embed(img_ids_padded)

        rope = (
            txt_cos.to(torch.bfloat16),
            txt_sin.to(torch.bfloat16),
            img_cos.to(torch.bfloat16),
            img_sin.to(torch.bfloat16),
        )
        self._rope_cache[cache_key] = rope
        return rope

    def _compute_rope_fallback(self, actual_txt_len):
        """Fallback RoPE computation when txt_ids/img_ids not provided."""
        cache_key = ("fallback", actual_txt_len)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids

        text_ids = prepare_pos_ids(
            modality_id=0, type="text", num_token=self.expected_txt_seq)
        target_ids = prepare_pos_ids(
            modality_id=1, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        source_ids = prepare_pos_ids(
            modality_id=2, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        img_ids = torch.cat([target_ids, source_ids], dim=0)

        return self._compute_rope_from_ids(text_ids, img_ids)

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, txt_ids=None, img_ids=None,
                return_dict=False, **kwargs):
        """
        Forward pass using compiled V3 CP transformer.

        hidden_states: [B, img_patches, 64] -- packed latents for target+source
        encoder_hidden_states: [B, txt_seq, 3584] -- text embeddings
        timestep: [B] -- denoising timestep
        txt_ids: [txt_seq, 3] -- text position IDs from pipeline
        img_ids: [img_seq, 3] -- image position IDs from pipeline
        """
        batch_size = hidden_states.shape[0]
        actual_txt_len = encoder_hidden_states.shape[1]

        # Compute RoPE from pipeline-provided position IDs
        if txt_ids is not None and img_ids is not None:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_from_ids(txt_ids, img_ids)
        else:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_fallback(actual_txt_len)

        # Pad hidden_states (image patches)
        actual_img = hidden_states.shape[1]
        if actual_img < self.expected_img_patches:
            pad = torch.zeros(
                (batch_size, self.expected_img_patches - actual_img, hidden_states.shape[2]),
                dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, pad], dim=1)
        elif actual_img > self.expected_img_patches:
            hidden_states = hidden_states[:, :self.expected_img_patches, :]

        # Pad encoder_hidden_states (text)
        if actual_txt_len < self.expected_txt_seq:
            pad = torch.zeros(
                (batch_size, self.expected_txt_seq - actual_txt_len, encoder_hidden_states.shape[2]),
                dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            encoder_hidden_states = torch.cat([encoder_hidden_states, pad], dim=1)
        elif actual_txt_len > self.expected_txt_seq:
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
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
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


class SimpleLatentDistribution:
    """Minimal latent distribution matching DiagonalGaussianDistribution interface."""
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        return self.mean  # Deterministic for compiled models


class SimpleEncoderOutput:
    """Minimal encoder output matching AutoencoderKLOutput interface."""
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class NeuronVAEWrapper:
    """
    Wrapper for compiled 2D AutoencoderKL matching the pipeline interface.

    IMPORTANT: Scaling (shift_factor, scaling_factor) is handled by the PIPELINE,
    NOT by this wrapper. The pipeline applies:
      encode: latents = (latents - shift_factor) * scaling_factor
      decode: latents = latents / scaling_factor + shift_factor

    This wrapper provides:
    - encode(x) -> returns object with .latent_dist.mode()/.sample()
    - decode(z, return_dict=False) -> returns (decoded_tensor,)
    - .config -> original VAE config (attribute-accessible)
    - Tiled processing for images larger than compiled tile size
    """
    def __init__(self, compiled_encoder, compiled_decoder, original_vae,
                 tile_h=512, tile_w=512):
        self.compiled_encoder = compiled_encoder
        self.compiled_decoder = compiled_decoder
        # Keep original VAE config for pipeline attribute access
        self.config = original_vae.config
        self.dtype = original_vae.dtype
        self.device = original_vae.device
        self.tile_h = tile_h
        self.tile_w = tile_w

    def encode(self, x, return_dict=True):
        """
        Encode image to latent space with tiled processing.

        Returns AutoencoderKLOutput-compatible object.
        Pipeline calls: retrieve_latents(self.vae.encode(image))
        which calls .latent_dist.mode() or .latent_dist.sample()
        """
        B, C, H, W = x.shape

        if H <= self.tile_h and W <= self.tile_w:
            moments = self.compiled_encoder(x)
        else:
            moments = self._tiled_encode(x)

        mean, logvar = torch.chunk(moments, 2, dim=1)
        dist = SimpleLatentDistribution(mean)

        if not return_dict:
            return (dist,)
        return SimpleEncoderOutput(dist)

    def decode(self, z, return_dict=False):
        """
        Decode latents to image with tiled processing.

        Pipeline calls: self.vae.decode(latents, return_dict=False)[0]
        """
        latent_h = z.shape[2]
        latent_w = z.shape[3]
        tile_latent_h = self.tile_h // 8
        tile_latent_w = self.tile_w // 8

        if latent_h <= tile_latent_h and latent_w <= tile_latent_w:
            decoded = self.compiled_decoder(z)
        else:
            decoded = self._tiled_decode(z)

        if return_dict:
            return type('DecoderOutput', (), {'sample': decoded})()
        return (decoded,)

    def _tiled_encode(self, x):
        """Tiled encoding for large images."""
        B, C, H, W = x.shape
        tile_h, tile_w = self.tile_h, self.tile_w

        latent_tiles = []
        for y in range(0, H, tile_h):
            row_tiles = []
            for x_start in range(0, W, tile_w):
                y_end = min(y + tile_h, H)
                x_end = min(x_start + tile_w, W)
                tile = x[:, :, y:y_end, x_start:x_end]

                if tile.shape[2] < tile_h or tile.shape[3] < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :tile.shape[2], :tile.shape[3]] = tile
                    tile = padded

                moments = self.compiled_encoder(tile)
                mean, logvar = torch.chunk(moments, 2, dim=1)
                row_tiles.append(mean[:, :, :((y_end - y) // 8), :((x_end - x_start) // 8)])
            latent_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in latent_tiles]
        full_mean = torch.cat(rows, dim=2)
        full_logvar = torch.zeros_like(full_mean)
        return torch.cat([full_mean, full_logvar], dim=1)

    def _tiled_decode(self, z):
        """Tiled decoding for large latents."""
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

                if tile.shape[2] < tile_h or tile.shape[3] < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :tile.shape[2], :tile.shape[3]] = tile
                    tile = padded

                decoded = self.compiled_decoder(tile)
                pixel_h = (y_end - y) * 8
                pixel_w = (x_end - x_start) * 8
                row_tiles.append(decoded[:, :, :pixel_h, :pixel_w])
            pixel_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in pixel_tiles]
        return torch.cat(rows, dim=2)


def load_transformer_v3_cp(compiled_models_dir, pipe, args):
    """Load V3 CP compiled transformer model."""
    v3_path = f"{compiled_models_dir}/transformer_v3_cp"
    nxd_model_path = f"{v3_path}/nxd_model.pt"
    weights_path = f"{v3_path}/weights"
    config_path = f"{v3_path}/config.json"

    for p, name in [(nxd_model_path, "model"), (weights_path, "weights"),
                     (config_path, "config")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"V3 CP {name} not found at {p}")

    with open(config_path, "r") as f:
        config = json.load(f)

    expected_img_patches = config["num_img_patches_padded"]
    expected_txt_seq = config["text_seq_len"]
    target_patches = config["num_img_patches"] // 2  # Only target image patches
    compiled_batch_size = config.get("batch_size", 1)
    patch_h = config["patch_h"]
    patch_w = config["patch_w"]

    print(f"  V3 CP config: img_patches={expected_img_patches}, txt_seq={expected_txt_seq}")
    print(f"  Target patches: {target_patches}, batch_size={compiled_batch_size}")
    print(f"  Patch grid: {patch_h}x{patch_w}")

    # Load NxDModel
    print(f"  Loading V3 CP model...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded weights
    # NxDModel expects one checkpoint per world_rank.
    # For CP: ranks within the same TP group share weights.
    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)

    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)
        print(f"    Loaded tp{rank}: {len(ckpt)} tensors")

    # Duplicate for all world ranks (CP ranks share TP weights)
    sharded_checkpoints = []
    for world_rank in range(world_size):
        tp_rank = world_rank % tp_degree
        sharded_checkpoints.append(tp_checkpoints[tp_rank])
    print(f"  Prepared {len(sharded_checkpoints)} weight shards for world_size={world_size}")

    nxd_model.set_weights(sharded_checkpoints)
    print("  Weights set, loading to Neuron...")
    nxd_model.to_neuron()
    print("  V3 CP model initialized on Neuron!")

    wrapper = NeuronTransformerWrapperV3CP(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        pos_embed=pipe.transformer.pos_embed,
        patch_h=patch_h,
        patch_w=patch_w,
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


def load_vae(compiled_models_dir, pipe, use_compiled=True):
    """Load compiled VAE or use original CPU VAE."""
    if not use_compiled:
        print("  Using original CPU VAE (compiled VAE skipped)")
        return pipe.vae

    encoder_path = f"{compiled_models_dir}/vae_encoder/model.pt"
    decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    config_path = f"{compiled_models_dir}/vae_config.json"

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("  WARNING: Compiled VAE not found, using CPU VAE")
        return pipe.vae

    with open(config_path) as f:
        vae_config = json.load(f)

    tile_h = vae_config.get("height", 512)
    tile_w = vae_config.get("width", 512)
    print(f"  Loading compiled VAE (tile: {tile_h}x{tile_w})")

    compiled_encoder = torch.jit.load(encoder_path)
    compiled_decoder = torch.jit.load(decoder_path)

    wrapper = NeuronVAEWrapper(
        compiled_encoder=compiled_encoder,
        compiled_decoder=compiled_decoder,
        original_vae=pipe.vae,
        tile_h=tile_h,
        tile_w=tile_w,
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
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--image_size", type=int, default=448, help="Vision encoder image size")
    parser.add_argument("--warmup", action="store_true", help="Run warmup inference")
    parser.add_argument("--skip_compiled_vae", action="store_true", help="Use CPU VAE instead of compiled")
    parser.add_argument("--skip_compiled_text_encoder", action="store_true",
                        help="Use CPU text encoder instead of compiled")
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR)
    parser.add_argument("--transformer_dir", type=str, default=None,
                        help="Override transformer compiled dir (default: <compiled_models_dir>)")
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
    transformer_dir = args.transformer_dir or args.compiled_models_dir
    print(f"Loading V3 CP transformer from {transformer_dir}...")
    neuron_transformer = load_transformer_v3_cp(transformer_dir, pipe, args)

    # Text encoder
    if args.skip_compiled_text_encoder:
        print("Using original CPU text encoder (compiled text encoder skipped)")
        neuron_text_encoder = pipe.text_encoder
    else:
        print("Loading text encoder...")
        neuron_text_encoder = load_text_encoder_v3(args.compiled_models_dir, pipe, args)

    # VAE
    print("Loading VAE...")
    neuron_vae = load_vae(args.compiled_models_dir, pipe, use_compiled=not args.skip_compiled_vae)

    # Replace pipeline components
    pipe.transformer = neuron_transformer
    pipe.text_encoder = neuron_text_encoder
    pipe.vae = neuron_vae

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
