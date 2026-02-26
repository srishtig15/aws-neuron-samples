"""
GPU reference script for LongCat-Image-Edit.

Use this to generate reference outputs for validating Neuron compilation.
Runs on a single CUDA GPU with standard diffusers pipeline.
"""

import os
import time
import torch
from PIL import Image
from diffusers import LongCatImageEditPipeline

# ===== Configuration =====
NUM_WARMUP = 1
NUM_RUNS = 3
NUM_INFERENCE_STEPS = 50

# ===== Load Model =====
load_start = time.perf_counter()
pipeline = LongCatImageEditPipeline.from_pretrained(
    "meituan-longcat/LongCat-Image-Edit",
    torch_dtype=torch.bfloat16,
)
load_end = time.perf_counter()
print(f"Pipeline loaded in {load_end - load_start:.2f}s")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=True)

# ===== Load Image =====
# Provide your own test image
IMAGE_PATH = "test_image.png"
if not os.path.exists(IMAGE_PATH):
    # Create a simple test image if none exists
    from PIL import Image
    test_img = Image.new('RGB', (512, 512), color=(100, 150, 200))
    test_img.save(IMAGE_PATH)
    print(f"Created test image: {IMAGE_PATH}")

source_image = Image.open(IMAGE_PATH).convert("RGB")
prompt = "Turn the image into a watercolor painting style"


def run_inference(seed=0):
    """Run one inference pass."""
    inputs = {
        "image": source_image,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]


# ===== Warmup =====
print(f"\n{'='*60}")
print(f"Warmup ({NUM_WARMUP} run(s))...")
print('='*60)

for i in range(NUM_WARMUP):
    warmup_start = time.perf_counter()
    torch.cuda.synchronize()
    _ = run_inference(seed=i)
    torch.cuda.synchronize()
    warmup_end = time.perf_counter()
    print(f"  Warmup {i+1}: {warmup_end - warmup_start:.2f}s")

torch.cuda.empty_cache()

# ===== Timed Runs =====
print(f"\n{'='*60}")
print(f"Timed runs ({NUM_RUNS} run(s))...")
print('='*60)

times = []
for i in range(NUM_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_image = run_inference(seed=100 + i)
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = end - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

# ===== Summary =====
print(f"\n{'='*60}")
print("Timing Summary")
print('='*60)
print(f"  Inference steps: {NUM_INFERENCE_STEPS}")
print(f"  Total runs: {NUM_RUNS}")
print(f"  Mean time: {sum(times)/len(times):.2f}s")
print(f"  Min time: {min(times):.2f}s")
print(f"  Max time: {max(times):.2f}s")
if len(times) > 1:
    import statistics
    print(f"  Std dev: {statistics.stdev(times):.2f}s")
print(f"  Throughput: {NUM_INFERENCE_STEPS / (sum(times)/len(times)):.2f} steps/s")

# ===== Save Output =====
output_image.save("output_longcat_gpu.png")
print(f"\nImage saved at {os.path.abspath('output_longcat_gpu.png')}")

# ===== GPU Memory =====
print(f"\n{'='*60}")
print("GPU Memory Usage")
print('='*60)
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
