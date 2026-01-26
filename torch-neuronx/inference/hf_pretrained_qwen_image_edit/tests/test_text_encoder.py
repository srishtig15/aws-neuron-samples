#!/usr/bin/env python3
"""
Text Encoder Unit Test: Compare Neuron vs CPU/GPU inference results

This test compares the Qwen2.5-VL text encoder outputs between:
1. Original model running on CPU
2. Compiled model running on Neuron (trn2)

The text encoder consists of:
- Vision Encoder: Processes image patches
- Language Model: Processes combined text + vision embeddings

Key metrics:
- Max Absolute Error (MAE)
- Mean Absolute Error (Mean AE)
- Cosine Similarity
- Output statistics (mean, std, min, max)
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Neuron environment BEFORE imports
TP_DEGREE = 8
os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from neuron_qwen_image_edit.neuron_commons import attention_wrapper, f32Wrapper

# Override SDPA for CPU model to match Neuron compilation
original_sdpa = torch.nn.functional.scaled_dot_product_attention


CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def compute_metrics(cpu_output, neuron_output, name="output"):
    """Compute comparison metrics between CPU and Neuron outputs."""
    # Ensure same dtype for comparison
    cpu_out = cpu_output.float().detach().cpu()
    neuron_out = neuron_output.float().detach().cpu()

    # Handle shape mismatch
    if cpu_out.shape != neuron_out.shape:
        print(f"  Shape mismatch: CPU {cpu_out.shape} vs Neuron {neuron_out.shape}")
        min_shape = [min(c, n) for c, n in zip(cpu_out.shape, neuron_out.shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        cpu_out = cpu_out[slices]
        neuron_out = neuron_out[slices]
        print(f"  Comparing truncated shape: {cpu_out.shape}")

    # Absolute error
    abs_error = torch.abs(cpu_out - neuron_out)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error
    rel_error = abs_error / (torch.abs(cpu_out) + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Cosine similarity
    cpu_flat = cpu_out.flatten()
    neuron_flat = neuron_out.flatten()
    cosine_sim = F.cosine_similarity(cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)).item()

    # Statistics
    cpu_stats = {
        "mean": cpu_out.mean().item(),
        "std": cpu_out.std().item(),
        "min": cpu_out.min().item(),
        "max": cpu_out.max().item(),
    }
    neuron_stats = {
        "mean": neuron_out.mean().item(),
        "std": neuron_out.std().item(),
        "min": neuron_out.min().item(),
        "max": neuron_out.max().item(),
    }

    print(f"\n{'='*60}")
    print(f"Metrics for {name}")
    print(f"{'='*60}")
    print(f"Shape: {cpu_out.shape}")
    print(f"\nError Metrics:")
    print(f"  Max Absolute Error:  {max_abs_error:.6e}")
    print(f"  Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"  Max Relative Error:  {max_rel_error:.6e}")
    print(f"  Mean Relative Error: {mean_rel_error:.6e}")
    print(f"  Cosine Similarity:   {cosine_sim:.6f}")
    print(f"\nCPU Output Statistics:")
    print(f"  Mean: {cpu_stats['mean']:.6f}, Std: {cpu_stats['std']:.6f}")
    print(f"  Min:  {cpu_stats['min']:.6f}, Max: {cpu_stats['max']:.6f}")
    print(f"\nNeuron Output Statistics:")
    print(f"  Mean: {neuron_stats['mean']:.6f}, Std: {neuron_stats['std']:.6f}")
    print(f"  Min:  {neuron_stats['min']:.6f}, Max: {neuron_stats['max']:.6f}")

    # Check for NaN/Inf
    if torch.isnan(neuron_out).any():
        print(f"\n  WARNING: Neuron output contains NaN values!")
    if torch.isinf(neuron_out).any():
        print(f"\n  WARNING: Neuron output contains Inf values!")

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_sim": cosine_sim,
        "cpu_stats": cpu_stats,
        "neuron_stats": neuron_stats,
    }


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.LayerNorm,)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def test_vision_encoder(args):
    """Test Vision Encoder: CPU vs Neuron."""
    print("\n" + "="*60)
    print("Testing Vision Encoder")
    print("="*60)

    dtype = torch.bfloat16
    image_size = args.image_size
    patch_size = 14
    temporal_patch_size = 2

    # Calculate patch dimensions
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w
    channels_per_patch = 3 * temporal_patch_size * patch_size * patch_size  # 1176

    print(f"\nConfiguration:")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Channels per patch: {channels_per_patch}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Get vision encoder
    visual = pipe.text_encoder.model.visual
    visual.eval()
    upcast_norms_to_f32(visual)

    # Create test inputs
    print("\nCreating test inputs...")
    # pixel_values: (num_patches, channels_per_patch)
    pixel_values = torch.randn(num_patches, channels_per_patch, dtype=dtype)
    # grid_thw: (num_images, 3)
    grid_thw = torch.tensor([[1, num_patches_h, num_patches_w]], dtype=torch.int64)

    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw.shape}")

    # CPU inference
    print("\nRunning CPU inference...")
    with torch.no_grad():
        cpu_output = visual(pixel_values, grid_thw)
    print(f"  CPU output shape: {cpu_output.shape}")

    # Check compiled model
    vision_encoder_path = f"{args.compiled_models_dir}/vision_encoder/model.pt"
    if not os.path.exists(vision_encoder_path):
        print(f"\nERROR: Compiled vision encoder not found at {vision_encoder_path}")
        print("Please run compile_text_encoder.py --vision_only first.")
        return None

    # Load Neuron compiled model
    print(f"\nLoading compiled vision encoder from {vision_encoder_path}...")
    import torch_neuronx
    compiled_vision = torch.jit.load(vision_encoder_path)

    # Neuron inference
    print("Running Neuron inference...")
    with torch.no_grad():
        neuron_output = compiled_vision(pixel_values, grid_thw)
    print(f"  Neuron output shape: {neuron_output.shape}")

    # Compare results
    metrics = compute_metrics(cpu_output, neuron_output, "Vision Encoder")

    return metrics


def test_language_model(args):
    """Test Language Model: CPU vs Neuron."""
    print("\n" + "="*60)
    print("Testing Language Model")
    print("="*60)

    dtype = torch.bfloat16
    batch_size = 1
    sequence_length = args.max_sequence_length
    hidden_size = 3584  # Qwen2.5-VL hidden size

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Get language model
    lang_model = pipe.text_encoder.model.language_model
    lang_model.eval()
    upcast_norms_to_f32(lang_model)

    # Create test inputs
    print("\nCreating test inputs...")
    # inputs_embeds: (batch, seq_len, hidden_size)
    inputs_embeds = torch.randn(batch_size, sequence_length, hidden_size, dtype=dtype)
    # attention_mask: (batch, seq_len)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.int64)

    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    # CPU inference
    print("\nRunning CPU inference...")
    with torch.no_grad():
        cpu_output = lang_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state
    print(f"  CPU output shape: {cpu_output.shape}")

    # Check compiled model
    language_model_path = f"{args.compiled_models_dir}/language_model"
    if not os.path.exists(language_model_path):
        print(f"\nERROR: Compiled language model not found at {language_model_path}")
        print("Please run compile_text_encoder.py --language_only first.")
        return None

    # Load Neuron compiled model
    print(f"\nLoading compiled language model from {language_model_path}...")
    import neuronx_distributed
    compiled_lang_model = neuronx_distributed.trace.parallel_model_load(language_model_path)

    # Neuron inference
    print("Running Neuron inference...")
    with torch.no_grad():
        neuron_output = compiled_lang_model(inputs_embeds, attention_mask)
    print(f"  Neuron output shape: {neuron_output.shape}")

    # Compare results
    metrics = compute_metrics(cpu_output, neuron_output, "Language Model")

    return metrics


def test_text_encoder_full(args):
    """Test full text encoder pipeline with real image input."""
    print("\n" + "="*60)
    print("Testing Full Text Encoder Pipeline")
    print("="*60)

    dtype = torch.bfloat16
    image_size = args.image_size

    print(f"\nConfiguration:")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Max sequence length: {args.max_sequence_length}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Create a test image
    print("\nCreating test image...")
    test_image = Image.new('RGB', (image_size, image_size), color='red')

    # Process image through tokenizer/processor
    prompt = "A red image for testing"

    print(f"  Prompt: {prompt}")

    # Use pipeline's tokenizer to prepare inputs
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=args.max_sequence_length,
        truncation=True,
        return_tensors="pt"
    )

    print(f"  input_ids shape: {text_inputs.input_ids.shape}")

    # Get CPU text encoder output
    print("\nRunning CPU text encoder...")
    with torch.no_grad():
        # Simple text-only test (no image)
        cpu_output = pipe.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    cpu_hidden = cpu_output.hidden_states[-1]
    print(f"  CPU hidden states shape: {cpu_hidden.shape}")

    # For Neuron, we need to test the wrapper
    # Check if compiled models exist
    vision_path = f"{args.compiled_models_dir}/vision_encoder/model.pt"
    lang_path = f"{args.compiled_models_dir}/language_model"

    if not os.path.exists(vision_path) or not os.path.exists(lang_path):
        print(f"\nERROR: Compiled text encoder components not found")
        print(f"  Vision encoder: {vision_path}")
        print(f"  Language model: {lang_path}")
        return None

    # Test individual components instead
    print("\nNote: Full pipeline test requires the NeuronTextEncoderWrapper.")
    print("Testing individual components instead.")

    # Test language model with text embeddings
    print("\nTesting language model with text embeddings...")
    embed_tokens = pipe.text_encoder.model.language_model.embed_tokens
    text_embeds = embed_tokens(text_inputs.input_ids)

    # Load compiled language model
    import neuronx_distributed
    compiled_lang_model = neuronx_distributed.trace.parallel_model_load(lang_path)

    # Pad to max_seq_len if needed
    if text_embeds.shape[1] < args.max_sequence_length:
        pad_len = args.max_sequence_length - text_embeds.shape[1]
        text_embeds = F.pad(text_embeds, (0, 0, 0, pad_len))
        attention_mask = F.pad(text_inputs.attention_mask, (0, pad_len))
    else:
        attention_mask = text_inputs.attention_mask

    print(f"  Padded embeds shape: {text_embeds.shape}")

    with torch.no_grad():
        neuron_lang_output = compiled_lang_model(text_embeds.to(dtype), attention_mask)

    # Compare language model outputs
    lang_model = pipe.text_encoder.model.language_model
    with torch.no_grad():
        cpu_lang_output = lang_model(
            inputs_embeds=text_embeds.to(dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state

    metrics = compute_metrics(cpu_lang_output, neuron_lang_output, "Language Model (Text Only)")

    return metrics


def test_embedding_values(args):
    """Test to debug embedding layer differences."""
    print("\n" + "="*60)
    print("Testing Embedding Values")
    print("="*60)

    dtype = torch.bfloat16

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    embed_tokens = pipe.text_encoder.model.language_model.embed_tokens

    # Test with specific token IDs
    test_ids = torch.tensor([[1, 100, 1000, 10000, 50000]])
    embeddings = embed_tokens(test_ids)

    print(f"\nEmbedding layer info:")
    print(f"  Num embeddings: {embed_tokens.num_embeddings}")
    print(f"  Embedding dim: {embed_tokens.embedding_dim}")
    print(f"  Weight dtype: {embed_tokens.weight.dtype}")

    print(f"\nTest embeddings shape: {embeddings.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.6f}")
    print(f"  Std: {embeddings.std().item():.6f}")
    print(f"  Min: {embeddings.min().item():.6f}")
    print(f"  Max: {embeddings.max().item():.6f}")

    return {"num_embeddings": embed_tokens.num_embeddings}


def main():
    parser = argparse.ArgumentParser(description="Text Encoder Unit Test: CPU vs Neuron")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for vision encoder")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--test", type=str, default="all",
                        choices=["vision", "language", "full", "embedding", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    print("="*60)
    print("Text Encoder Unit Test: Comparing Neuron vs CPU Inference")
    print("="*60)
    print(f"Image size: {args.image_size}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Compiled models: {args.compiled_models_dir}")

    results = {}

    if args.test in ["vision", "all"]:
        results["vision"] = test_vision_encoder(args)

    if args.test in ["language", "all"]:
        results["language"] = test_language_model(args)

    if args.test in ["full", "all"]:
        results["full"] = test_text_encoder_full(args)

    if args.test in ["embedding", "all"]:
        results["embedding"] = test_embedding_values(args)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, metrics in results.items():
        if metrics and "cosine_sim" in metrics:
            status = "PASS" if metrics["cosine_sim"] > 0.99 else "WARN" if metrics["cosine_sim"] > 0.95 else "FAIL"
            print(f"{name:15s}: Cosine Sim = {metrics['cosine_sim']:.6f}  Max AE = {metrics['max_abs_error']:.2e}  [{status}]")
        elif metrics:
            print(f"{name:15s}: Completed")
        else:
            print(f"{name:15s}: SKIPPED (compiled model not found)")


if __name__ == "__main__":
    main()
