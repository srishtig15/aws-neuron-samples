"""
Download and cache model weights for RollingForcing Neuron inference.

Downloads:
1. Wan-AI/Wan2.1-T2V-1.3B-Diffusers (text encoder, VAE, base transformer config)
2. TencentARC/RollingForcing DMD checkpoint
"""
import os
import argparse
import subprocess
import sys


def ensure_dependencies():
    """Install required Python packages."""
    packages = [
        "diffusers",
        "transformers",
        "omegaconf",
        "easydict",
        "einops",
        "safetensors",
        "accelerate",
        "sentencepiece",
        "protobuf",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def download_wan21_model(cache_dir):
    """Download Wan2.1-T2V-1.3B from HuggingFace."""
    from diffusers import AutoencoderKLWan
    from transformers import UMT5EncoderModel, AutoTokenizer
    import torch

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    print(f"\nDownloading {model_id}...")

    # Text encoder (UMT5-XXL)
    print("  Downloading text encoder (UMT5-XXL)...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    del text_encoder

    # Tokenizer
    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer",
        cache_dir=cache_dir,
    )
    del tokenizer

    # VAE
    print("  Downloading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )
    del vae

    print(f"  Wan2.1-T2V-1.3B downloaded to {cache_dir}")


def download_rolling_forcing_checkpoint(cache_dir):
    """Download RollingForcing DMD checkpoint."""
    from huggingface_hub import hf_hub_download

    repo_id = "TencentARC/RollingForcing"
    filename = "checkpoints/rolling_forcing_dmd.pt"

    ckpt_dir = os.path.join(cache_dir, "rolling_forcing")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, filename)
    if os.path.exists(ckpt_path):
        print(f"\n  RollingForcing checkpoint already exists at {ckpt_path}")
        return ckpt_path

    print(f"\nDownloading RollingForcing checkpoint from {repo_id}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        local_dir=ckpt_dir,
    )
    print(f"  RollingForcing checkpoint downloaded to {downloaded_path}")
    return downloaded_path


def main():
    parser = argparse.ArgumentParser(description="Download model weights for RollingForcing")
    parser.add_argument(
        "--cache_dir", type=str,
        default="/opt/dlami/nvme/rolling_forcing_hf_cache",
        help="Directory to cache downloaded models",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 60)
    print("RollingForcing Model Download")
    print("=" * 60)

    ensure_dependencies()
    download_wan21_model(args.cache_dir)
    download_rolling_forcing_checkpoint(args.cache_dir)

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
