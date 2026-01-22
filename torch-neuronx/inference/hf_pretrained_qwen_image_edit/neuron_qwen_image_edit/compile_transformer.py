import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # For trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # For trn2

compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import neuronx_distributed
from functools import partial
from torch import nn

from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from neuron_commons import attention_wrapper_for_transformer
from neuron_parallel_utils import shard_qwen_attention, shard_feedforward

# Note: Do NOT override SDPA globally during compilation
# The diffusers attention processor handles attention internally
# torch.nn.functional.scaled_dot_product_attention = attention_wrapper_for_transformer

CACHE_DIR = "qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


class TracingTransformerWrapper(nn.Module):
    """Wrapper for tracing the transformer model."""
    def __init__(self, transformer: QwenImageTransformer2DModel, img_shapes):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        # Store img_shapes as a fixed attribute for tracing
        self.img_shapes = img_shapes

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        """
        Forward pass matching QwenImageTransformer2DModel signature.

        Args:
            hidden_states: (batch, num_patches, in_channels) - patchified latents
            encoder_hidden_states: (batch, text_seq_len, text_hidden_dim) - text embeddings
            timestep: (batch,) - diffusion timestep
        """
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=self.img_shapes,
            return_dict=False)


def get_transformer_model(tp_degree: int, img_shapes: list):
    """Load and shard the transformer model for tensor parallelism."""
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=CACHE_DIR)

    # Shard transformer blocks
    for block_idx, block in enumerate(pipe.transformer.transformer_blocks):
        # Shard attention
        block.attn = shard_qwen_attention(tp_degree, block.attn)
        # Shard feedforward (img_mlp and txt_mlp)
        block.img_mlp = shard_feedforward(block.img_mlp)
        block.txt_mlp = shard_feedforward(block.txt_mlp)

    transformer_wrapper = TracingTransformerWrapper(pipe.transformer, img_shapes)
    return transformer_wrapper, {}


def compile_transformer(args):
    tp_degree = 4  # Tensor parallel degree for trn2
    os.environ["LOCAL_WORLD_SIZE"] = "4"

    latent_height = args.height // 8
    latent_width = args.width // 8
    max_sequence_length = args.max_sequence_length
    text_hidden_size = 3584  # Text encoder hidden size
    in_channels = 64  # QwenImage transformer in_channels
    patch_size = 2  # QwenImage patch size
    temporal_frames = 1  # For single image

    # Calculate number of patches
    # QwenImage uses patch_size=2, so num_patches = T * (H/8/2) * (W/8/2)
    num_patches = temporal_frames * (latent_height // patch_size) * (latent_width // patch_size)

    # img_shapes: List of (frame, height, width) for each batch item
    # Note: height/width here are in patch space (latent_h // patch_size)
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    img_shapes = [(temporal_frames, patch_h, patch_w)] * 2  # batch_size=2

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    batch_size = 2  # For CFG (conditional + unconditional)

    print(f"Compiling transformer with:")
    print(f"  Image size: {args.height}x{args.width}")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Text sequence length: {max_sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  img_shapes: {img_shapes}")

    # Sample inputs matching transformer wrapper forward signature
    # hidden_states: (batch, num_patches, in_channels)
    sample_hidden_states = torch.ones(
        (batch_size, num_patches, in_channels), dtype=torch.bfloat16)
    # encoder_hidden_states: (batch, text_seq_len, text_hidden_size)
    sample_encoder_hidden_states = torch.ones(
        (batch_size, max_sequence_length, text_hidden_size), dtype=torch.bfloat16)
    # timestep: (batch,)
    sample_timestep = torch.ones((batch_size,), dtype=torch.float32)

    get_transformer_f = partial(get_transformer_model, tp_degree, img_shapes)

    with torch.no_grad():
        sample_inputs = (
            sample_hidden_states,
            sample_encoder_hidden_states,
            sample_timestep,
        )

        compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
            get_transformer_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/transformer",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )

        compiled_model_dir = f"{compiled_models_dir}/transformer"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)

        neuronx_distributed.trace.parallel_model_save(
            compiled_transformer, compiled_model_dir)
        print(f"Transformer compiled and saved to {compiled_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024,
                        help="Height of generated image")
    parser.add_argument("--width", type=int, default=1024,
                        help="Width of generated image")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max sequence length for text encoder")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir",
                        help="Directory for compiler artifacts")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models",
                        help="Directory for compiled models")
    args = parser.parse_args()
    compile_transformer(args)
