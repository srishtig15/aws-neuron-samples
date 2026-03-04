"""
Download and cache Wan2.2-T2V-A14B model weights (~126GB).

Downloads the full pipeline including both transformers (high-noise and low-noise experts),
VAE, text encoder, and tokenizer.
"""
import os
import argparse
import torch
from diffusers import AutoencoderKLWan, WanPipeline


def main():
    parser = argparse.ArgumentParser(description="Download Wan2.2-T2V-A14B model")
    parser.add_argument(
        "--cache_dir", type=str,
        default="/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir",
        help="Directory to cache downloaded models",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    DTYPE = torch.bfloat16

    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print(f"Downloading {model_id}")
    print(f"Cache dir: {cache_dir}")
    print("=" * 60)

    print("\nDownloading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )
    del vae

    print("\nDownloading full pipeline (transformers, text encoder, tokenizer)...")
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        cache_dir=cache_dir,
    )
    del pipe

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
