"""
Language Model Compilation using ModelBuilder API (V3) for V3 CP Compatibility.

Compiles the Qwen2.5-VL Language Model (shared between Qwen-Image-Edit and
LongCat-Image-Edit) using ModelBuilder API with tp_degree=4 and world_size=8.

Key features:
- TP=4 is perfect for Qwen2.5-VL GQA: 28Q/4=7 heads/rank, 4KV/4=1 head/rank
- world_size=8 for compatibility with V3 CP transformer
- No Context Parallel needed (language model processes full sequence)

Usage:
    neuron_parallel_compile python compile_language_model_v3.py --max_sequence_length 512
"""

import os
import json
import gc

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse

from diffusers import LongCatImageEditPipeline

from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state

from neuron_parallel_utils import shard_qwen2_attention, shard_qwen2_mlp, get_sharded_data

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"


def load_pipeline(dtype=torch.bfloat16):
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x, *args, **kwargs):
        t = x.dtype
        output = self.original(x.to(torch.float32), *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


class NeuronLanguageModelV3(nn.Module):
    """Neuron-optimized Qwen2.5-VL Language Model with TP=4."""

    def __init__(self, original_language_model, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.language_model = original_language_model
        self.config = original_language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers

        print(f"  Language model: hidden_size={self.hidden_size}, layers={self.num_hidden_layers}")
        print(f"    Q heads: {self.config.num_attention_heads}, KV heads: {self.config.num_key_value_heads}")

        for i, layer in enumerate(self.language_model.layers):
            layer.self_attn = shard_qwen2_attention(tp_degree, layer.self_attn)
            layer.mlp = shard_qwen2_mlp(layer.mlp)
            if i == 0:
                print(f"  Sharded layer 0")
        print(f"  Sharded all {len(self.language_model.layers)} layers")

        upcast_norms_to_f32(self.language_model)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TracingWrapper(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
    def forward(self, inputs_embeds, attention_mask, position_ids):
        return self.language_model(inputs_embeds, attention_mask, position_ids)


def compile_language_model_v3(args):
    tp_degree = 4
    world_size = 8
    batch_size = args.batch_size
    sequence_length = args.max_sequence_length
    hidden_size = 3584

    print("=" * 60)
    print("Compiling Language Model V3 (ModelBuilder API)")
    print("=" * 60)
    print(f"  Batch={batch_size}, SeqLen={sequence_length}, TP={tp_degree}, World={world_size}")

    sample_inputs_embeds = torch.randn(batch_size, sequence_length, hidden_size, dtype=torch.bfloat16)
    sample_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.int64)
    sample_position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1).clone()

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("Loading model...")
        pipe = load_pipeline(torch.bfloat16)

        original_language_model = pipe.text_encoder.model.language_model
        unsharded_state = original_language_model.state_dict()

        print(f"\nCreating Neuron language model (TP={tp_degree})...")
        neuron_lm = NeuronLanguageModelV3(original_language_model, tp_degree)
        neuron_lm = neuron_lm.to(torch.bfloat16)
        neuron_lm.eval()

        del pipe
        gc.collect()

        model = TracingWrapper(neuron_lm)

        builder = ModelBuilder(model=model)
        print("Tracing...")
        builder.trace(
            kwargs={
                "inputs_embeds": sample_inputs_embeds,
                "attention_mask": sample_attention_mask,
                "position_ids": sample_position_ids,
            },
            tag="inference",
        )

        print("Compiling...")
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=args.compiler_workdir,
        )

        output_path = f"{args.compiled_models_dir}/language_model_v3"
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        checkpoint = {}
        for key, value in model.state_dict().items():
            orig_key = key.replace("language_model.language_model.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        shard_checkpoint(checkpoint=checkpoint, model=model, serialize_path=weights_path)

        # Post-process
        from safetensors.torch import load_file, save_file
        inv_freq_buffers = {}
        for name, buf in neuron_lm.language_model.named_buffers():
            if 'inv_freq' in name:
                inv_freq_buffers[f"language_model.language_model.{name}"] = buf.to(torch.bfloat16).clone()
        print(f"  Collected {len(inv_freq_buffers)} inv_freq buffers")

        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                continue
            data = dict(load_file(shard_file))
            cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}
            cleaned.update(inv_freq_buffers)
            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {len(data)} -> {len(cleaned)} tensors")

        config = {
            "max_sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "tp_degree": tp_degree,
            "world_size": world_size,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nLanguage Model V3 compiled: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    args = parser.parse_args()

    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_language_model_v3(args)
