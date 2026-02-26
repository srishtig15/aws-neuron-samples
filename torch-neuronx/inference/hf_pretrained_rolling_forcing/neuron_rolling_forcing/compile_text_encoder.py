"""
Compile UMT5-XXL text encoder for RollingForcing on Neuron.

Reuses the same UMT5 architecture from Wan2.2 (text_dim=4096, 64 heads, 24 layers).
Adapted for Wan2.1-T2V-1.3B model ID.

TP=8, bf16, pre-computed attention bias per TP rank.
"""
import os

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = " --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
import torch_neuronx
import neuronx_distributed
from transformers.models.umt5 import UMT5EncoderModel
from torch import nn
from functools import partial

from transformers.models.umt5.modeling_umt5 import UMT5Block, UMT5LayerSelfAttention, UMT5LayerFF

from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import get_sharded_data, shard_umt5_self_attention, shard_umt5_ff

# Replace SDPA with Neuron-compatible version
torch.nn.functional.scaled_dot_product_attention = attention_wrapper


class TracingUMT5WrapperTP(nn.Module):
    """Wrapper for UMT5 encoder with TP and pre-computed attention bias."""
    def __init__(self, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.t = t
        self.device = t.device
        for block_idx in range(len(self.t.encoder.block)):
            precomputed_bias = self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
            precomputed_bias_tp = get_sharded_data(precomputed_bias, 1)
            self.t.encoder.block[block_idx].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp

    def forward(self, text_input_ids, attention_mask=None):
        return self.t(text_input_ids, attention_mask=attention_mask)


def get_text_encoder(tp_degree: int, sequence_length: int, cache_dir: str):
    """Load and shard text encoder for Neuron compilation."""
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DTYPE = torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder",
        torch_dtype=DTYPE, cache_dir=cache_dir,
    )
    text_encoder.eval()

    for idx, block in enumerate(text_encoder.encoder.block):
        block: UMT5Block = block
        selfAttention: UMT5LayerSelfAttention = block.layer[0].SelfAttention
        ff: UMT5LayerFF = block.layer[1]
        layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
        layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)
        block.layer[1] = shard_umt5_ff(ff)
        block.layer[0].SelfAttention = shard_umt5_self_attention(tp_degree, selfAttention)
        block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
        block.layer[1].layer_norm = f32Wrapper(layer_norm_1)

    final_layer_norm = text_encoder.encoder.final_layer_norm.to(torch.float32)
    text_encoder.encoder.final_layer_norm = f32Wrapper(final_layer_norm)

    return TracingUMT5WrapperTP(text_encoder, sequence_length), {}


def compile_text_encoder(args):
    """Compile text encoder with tensor parallelism."""
    batch_size = 1
    sequence_length = args.max_sequence_length
    tp_degree = args.tp_degree
    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    get_text_encoder_f = partial(
        get_text_encoder, tp_degree, sequence_length, args.cache_dir)

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir

    print("=" * 60)
    print("RollingForcing Text Encoder (UMT5-XXL) Compilation")
    print("=" * 60)
    print(f"Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    print(f"Sequence length: {sequence_length}")
    print(f"TP degree: {tp_degree}")
    print("=" * 60)

    with torch.no_grad():
        sample_inputs = (
            torch.ones((batch_size, sequence_length), dtype=torch.int64),
            torch.ones((batch_size, sequence_length), dtype=torch.int64),
        )
        compiled_text_encoder = neuronx_distributed.trace.parallel_model_trace(
            get_text_encoder_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/text_encoder",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )
        compiled_model_dir = f"{compiled_models_dir}/text_encoder"
        os.makedirs(compiled_model_dir, exist_ok=True)
        neuronx_distributed.trace.parallel_model_save(
            compiled_text_encoder, compiled_model_dir)

    print(f"\nText encoder compiled and saved to {compiled_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--cache_dir", type=str,
                       default="/opt/dlami/nvme/rolling_forcing_hf_cache")
    args = parser.parse_args()
    compile_text_encoder(args)
