"""
Compile QwenImage Transformer with NKI Flash Attention for Neuron (TRN2).

This version uses NKI Flash Attention for significantly improved performance.
Requirements:
- Sequence length must be divisible by 2048
- For 1024x1024 with patch_multiplier=3: 12288 patches + 512 text = 12800 (needs padding to 14336)
"""

import os

# Environment setup for TRN2
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags for TRN2 with flash attention optimization
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import neuronx_distributed
from functools import partial
from torch import nn

from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from neuron_parallel_utils import shard_qwen_attention, shard_feedforward, shard_modulation
from neuron_rope import patch_qwenimage_rope
from neuron_flash_attention import (
    patch_transformer_with_flash_attention,
    calculate_total_seq_len,
    validate_sequence_length
)

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


class TracingTransformerWrapper(nn.Module):
    """Wrapper for tracing the transformer model with flash attention."""

    def __init__(self, transformer: QwenImageTransformer2DModel, img_shapes):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        self.img_shapes = img_shapes

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=self.img_shapes,
            return_dict=False
        )


def get_transformer_model(tp_degree: int, img_shapes: list, use_flash_attention: bool = True):
    """Load, patch with flash attention, and shard the transformer model."""

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=CACHE_DIR
    )

    # 1. Patch RoPE for Neuron compatibility
    print("Patching RoPE for Neuron compatibility...")
    pipe.transformer = patch_qwenimage_rope(pipe.transformer)

    # 2. Patch with NKI Flash Attention
    if use_flash_attention:
        print("Patching transformer with NKI Flash Attention...")
        pipe.transformer = patch_transformer_with_flash_attention(
            pipe.transformer,
            use_flash_attention=True,
            lnc=2  # TRN2 uses lnc=2
        )

    # 3. Shard transformer blocks for tensor parallelism
    num_blocks = len(pipe.transformer.transformer_blocks)
    print(f"Sharding {num_blocks} transformer blocks with TP={tp_degree}")

    for block_idx, block in enumerate(pipe.transformer.transformer_blocks):
        if block_idx == 0:
            print(f"Block 0 attention heads: {block.attn.heads}")
            print(f"Block 0 to_q shape: {block.attn.to_q.weight.shape}")

        # Shard attention
        block.attn = shard_qwen_attention(tp_degree, block.attn)

        # Shard feedforward
        block.img_mlp = shard_feedforward(block.img_mlp)
        block.txt_mlp = shard_feedforward(block.txt_mlp)

        # Shard modulation layers
        block.img_mod = shard_modulation(block.img_mod)
        block.txt_mod = shard_modulation(block.txt_mod)

        if (block_idx + 1) % 10 == 0:
            print(f"  Processed {block_idx + 1}/{num_blocks} blocks")

    print(f"All {num_blocks} blocks sharded successfully")

    transformer_wrapper = TracingTransformerWrapper(pipe.transformer, img_shapes)
    return transformer_wrapper, {}


def compile_transformer(args):
    tp_degree = args.tp_degree
    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    latent_height = args.height // 8
    latent_width = args.width // 8
    max_sequence_length = args.max_sequence_length
    text_hidden_size = 3584
    in_channels = 64
    patch_size = 2

    temporal_frames = args.patch_multiplier
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    num_patches = temporal_frames * patch_h * patch_w

    # Calculate and validate total sequence length
    total_seq_len = calculate_total_seq_len(
        args.height, args.width, args.patch_multiplier, max_sequence_length
    )
    validate_sequence_length(total_seq_len)

    img_shapes = [(temporal_frames, patch_h, patch_w)] * args.batch_size

    print(f"\n{'='*60}")
    print("Compiling Transformer with NKI Flash Attention")
    print('='*60)
    print(f"  Image size: {args.height}x{args.width}")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Text sequence length: {max_sequence_length}")
    print(f"  Total sequence length: {total_seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  TP degree: {tp_degree}")
    print(f"  Use Flash Attention: {args.use_flash_attention}")
    print(f"  img_shapes: {img_shapes}")

    # Sample inputs
    sample_hidden_states = torch.ones(
        (args.batch_size, num_patches, in_channels), dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.ones(
        (args.batch_size, max_sequence_length, text_hidden_size), dtype=torch.bfloat16)
    sample_timestep = torch.ones((args.batch_size,), dtype=torch.float32)

    get_transformer_f = partial(
        get_transformer_model, tp_degree, img_shapes, args.use_flash_attention
    )

    with torch.no_grad():
        sample_inputs = (
            sample_hidden_states,
            sample_encoder_hidden_states,
            sample_timestep,
        )

        print("\nStarting compilation (this may take a while)...")
        compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
            get_transformer_f,
            sample_inputs,
            compiler_workdir=f"{args.compiler_workdir}/transformer_flash",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )

        # Save to a different directory to distinguish from non-flash version
        compiled_model_dir = f"{args.compiled_models_dir}/transformer_flash"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)

        neuronx_distributed.trace.parallel_model_save(
            compiled_transformer, compiled_model_dir)
        print(f"\nTransformer (Flash Attention) compiled and saved to {compiled_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile transformer with NKI Flash Attention")
    parser.add_argument("--height", type=int, default=1024,
                        help="Height of generated image")
    parser.add_argument("--width", type=int, default=1024,
                        help="Width of generated image")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                        help="Max sequence length for text encoder (2048 recommended for Flash Attention alignment)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--tp_degree", type=int, default=8,
                        help="Tensor parallel degree")
    parser.add_argument("--patch_multiplier", type=int, default=3,
                        help="Patch multiplier (3 for 2-image merge, 2 for editing)")
    parser.add_argument("--use_flash_attention", type=bool, default=True,
                        help="Use NKI Flash Attention")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir",
                        help="Directory for compiler artifacts")
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models",
                        help="Directory for compiled models")
    args = parser.parse_args()

    compile_transformer(args)
