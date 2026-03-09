"""Wan2.2-T2V-A14B GPU inference benchmark.

Aligns with official Wan2.2 repo and HuggingFace model card parameters:
  - guidance_scale=4.0, guidance_scale_2=3.0
  - num_inference_steps=40, fps=16
  - Supported resolutions: 480P (480x832), 720P (720x1280)

Usage:
  python run_wan2.2_t2v_a14b_gpu.py                          # 480P default
  python run_wan2.2_t2v_a14b_gpu.py --resolution 720P        # 720P
  python run_wan2.2_t2v_a14b_gpu.py --offload                # CPU offload (saves VRAM)
  python run_wan2.2_t2v_a14b_gpu.py --prompt "your prompt"

Flash-attn-4 support (optional, for CUDA 13 + Hopper GPUs):
  pip install --pre flash-attn-4
  python patch_diffusers_fa4.py
  DIFFUSERS_ATTN_BACKEND="_flash_4" python run_wan2.2_t2v_a14b_gpu.py
"""
import argparse
import os
import torch
import time
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

RESOLUTION_MAP = {
    "480P": (480, 832),
    "720P": (720, 1280),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2-T2V-A14B GPU benchmark")
    parser.add_argument("--resolution", type=str, default="480P", choices=["480P", "720P"],
                        help="Video resolution (default: 480P)")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of video frames, must be 4n+1 (default: 81)")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Number of denoising steps (default: 40)")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="High-noise guidance scale (default: 4.0)")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0,
                        help="Low-noise guidance scale (default: 3.0)")
    parser.add_argument("--prompt", type=str,
                        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
                        help="Text prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, misshapen limbs, fused fingers, still image, messy background, three legs, many people in background, walking backwards",
                        help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS (default: 16)")
    parser.add_argument("--cache_dir", type=str, default="/opt/dlami/nvme/hf_cache",
                        help="HuggingFace model cache directory")
    parser.add_argument("--offload", action="store_true",
                        help="Enable model CPU offload (reduces VRAM, slower)")
    parser.add_argument("--no_warmup", action="store_true",
                        help="Skip warmup run")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: output_t2v_a14b_{resolution}.mp4)")
    return parser.parse_args()


def main():
    args = parse_args()
    height, width = RESOLUTION_MAP[args.resolution]
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    if args.output is None:
        args.output = f"output_t2v_a14b_{args.resolution.lower()}.mp4"

    # Report attention backend
    attn_backend = os.environ.get("DIFFUSERS_ATTN_BACKEND", "native (PyTorch SDPA)")
    print(f"Resolution: {args.resolution} ({height}x{width})")
    print(f"Frames: {args.num_frames}, Steps: {args.num_inference_steps}")
    print(f"Guidance: {args.guidance_scale} (high-noise), {args.guidance_scale_2} (low-noise)")
    print(f"Offload: {args.offload}")
    print(f"Attention backend: {attn_backend}")

    print("\nLoading pipeline...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=args.cache_dir
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    )
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    print(f"Pipeline loaded in {time.time()-t0:.1f}s")

    gen_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        max_sequence_length=512,
    )

    if not args.no_warmup:
        print("\nWarmup...")
        t0 = time.time()
        _ = pipe(**gen_kwargs).frames[0]
        print(f"Warmup done in {time.time()-t0:.1f}s")

    print("\nBenchmark...")
    torch.manual_seed(args.seed)
    t0 = time.time()
    output = pipe(**gen_kwargs).frames[0]
    total = time.time() - t0
    print(f"\nTotal inference: {total:.1f}s")

    export_to_video(output, args.output, fps=args.fps)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
