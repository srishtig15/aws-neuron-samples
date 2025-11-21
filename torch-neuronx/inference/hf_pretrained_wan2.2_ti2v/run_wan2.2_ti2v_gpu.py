"""
GPU版本的推理脚本，用于与Trainium结果对比

使用方法：
    python run_wan2.2_ti2v_gpu.py

注意：
    - 使用与Trainium版本相同的随机种子（SEED=42）
    - 使用相同的prompt、参数和配置
    - 生成的output_gpu.mp4可与Trainium的output.mp4对比
"""

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import numpy as npy
import os
import random
import time
import torch


def set_seed(seed: int):
    """
    设置所有随机种子以确保结果可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    npy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置确定性行为（可能会影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")


HUGGINGFACE_CACHE_DIR = "wan2.2_ti2v_hf_cache_dir"
SEED = 42  # 与Trainium版本相同的随机种子

if __name__ == "__main__":
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("错误: 未检测到GPU。请在有GPU的机器上运行此脚本。")
        exit(1)

    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # 设置随机种子以确保结果可复现
    set_seed(SEED)

    # 创建PyTorch生成器用于diffusion采样
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    print("加载模型...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=DTYPE,
        cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = pipe.to("cuda")

    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    seqlen = 512

    print("\n开始 warmup 推理...")
    start = time.time()
    output_warmup = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=50,
        max_sequence_length=seqlen,
        generator=torch.Generator(device="cuda").manual_seed(SEED + 1000)  # warmup使用不同的种子
    ).frames[0]
    end = time.time()
    print(f'warmup time: {end-start:.2f}s')

    # 重置generator以确保主推理使用一致的随机种子
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    print("\n开始主推理（使用固定种子）...")
    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=50,
        max_sequence_length=seqlen,
        generator=generator  # 使用固定种子的generator
    ).frames[0]
    end = time.time()

    print(f'time: {end-start:.2f}s')
    print(f"Output shape: {output.shape}")
    print(f"Output frames: {len(output)}")

    output_path = "output_gpu.mp4"
    export_to_video(output, output_path, fps=24)
    print(f"\n视频已保存到: {output_path}")

    # 可选: 保存numpy数组用于数值对比
    # 取消下面的注释以保存numpy数组
    # npy.save("output_gpu.npy", output)
    # print("Numpy数组已保存到: output_gpu.npy")

    print("\n对比方法：")
    print(f"  1. 视觉对比: 直接播放 output_gpu.mp4 和 output.mp4")
    print(f"  2. 数值对比: 取消脚本中npy.save的注释，然后运行:")
    print(f"     python compare_outputs.py output_gpu.npy output_trainium.npy")
    print(f"     - GPU输出shape: {output.shape}")
    print(f"     - Trainium输出应该有相同的shape")
