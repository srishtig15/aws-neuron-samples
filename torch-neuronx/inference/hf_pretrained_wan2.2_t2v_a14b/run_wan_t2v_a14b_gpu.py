"""Wan2.2-T2V-A14B GPU inference benchmark."""
import torch
import time
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CACHE_DIR = "/opt/dlami/nvme/hf_cache"
PROMPT = "A cat walks on the grass, realistic"
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles"

print("Loading pipeline...")
t0 = time.time()
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR)
pipe = pipe.to("cuda")
print(f"Pipeline loaded in {time.time()-t0:.1f}s")

# Optional: enable if OOM
# pipe.enable_model_cpu_offload()

print("\nWarmup...")
t0 = time.time()
_ = pipe(
    prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
    height=480, width=832, num_frames=81,
    num_inference_steps=50, guidance_scale=5.0,
    max_sequence_length=512,
).frames[0]
print(f"Warmup done in {time.time()-t0:.1f}s")

print("\nBenchmark...")
torch.manual_seed(42)
t0 = time.time()
output = pipe(
    prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
    height=480, width=832, num_frames=81,
    num_inference_steps=50, guidance_scale=5.0,
    max_sequence_length=512,
).frames[0]
total = time.time() - t0
print(f"\nTotal inference: {total:.1f}s")

export_to_video(output, "output_t2v_a14b_gpu.mp4", fps=24)
print("Saved output_t2v_a14b_gpu.mp4")
