import torch
from diffusers import AutoencoderKLWan, WanPipeline

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DTYPE = torch.bfloat16
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.2_ti2v_hf_cache_dir")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.2_ti2v_hf_cache_dir")
