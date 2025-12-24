# imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import neuronx_distributed
import numpy as npy
import os
import random
import time
import torch
import torch_neuronx

from neuron_wan2_2_ti2v.neuron_commons import InferenceTextEncoderWrapper
from neuron_wan2_2_ti2v.neuron_commons import InferenceTransformerWrapper
from neuron_wan2_2_ti2v.neuron_commons import SimpleWrapper
from neuron_wan2_2_ti2v.neuron_commons import DecoderWrapper


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

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.2_ti2v_hf_cache_dir"
SEED = 42  # 固定随机种子，用于对比GPU和Trainium的结果

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    set_seed(SEED)

    # 创建PyTorch生成器用于diffusion采样
    generator = torch.Generator().manual_seed(SEED)

    DTYPE=torch.bfloat16
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR)
    # print(pipe.text_encoder)
    # print(pipe.transformer)
    # print(pipe.vae)
        
    text_encoder_model_path = f"{COMPILED_MODELS_DIR}/text_encoder"
    transformer_model_path = f"{COMPILED_MODELS_DIR}/transformer" 
    decoder_model_path = f"{COMPILED_MODELS_DIR}/decoder/model.pt"
    post_quant_conv_model_path = f"{COMPILED_MODELS_DIR}/post_quant_conv/model.pt"
    
    seqlen=512  # default: 512
    
    text_encoder_wrapper = InferenceTextEncoderWrapper(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    print('text_encoder_wrapper.t start ****************')
    text_encoder_wrapper.t = neuronx_distributed.trace.parallel_model_load(
        text_encoder_model_path
    )
    # text_encoder_wrapper.t = torch_neuronx.DataParallel( 
    #     # torch.jit.load(os.path.join(text_encoder_model_path, 'model.pt')), [0, 1, 2, 3], False  # Use for trn2
    #     torch.jit.load(os.path.join(text_encoder_model_path, 'model.pt')), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    # )
    print('text_encoder_wrapper.t end ****************')

    transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
    print('transformer_wrapper.transformer start ****************')
    transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(
        transformer_model_path
    )
    # transformer_wrapper.transformer = torch_neuronx.DataParallel( 
    #     torch.jit.load(os.path.join(transformer_model_path, 'model.pt')), [0, 1, 2, 3], False  # Use for trn2
    #     # torch.jit.load(os.path.join(transformer_model_path, 'model.pt')), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    # )
    print('transformer_wrapper.transformer end ****************')

    vae_decoder_wrapper = DecoderWrapper(pipe.vae.decoder)
    print('vae_decoder_wrapper.model start ****************')
    # Decoder CANNOT use DataParallel because it accepts feat_cache (list argument)
    # DataParallel's scatter mechanism cannot handle list of tensors
    # Use DecoderWrapper to handle TorchScript's feat_cache compatibility (None -> zero tensors)
    vae_decoder_wrapper.model = torch.jit.load(decoder_model_path)
    # print('vae_decoder_wrapper.model:', vae_decoder_wrapper.model)
    print('vae_decoder_wrapper.model end ****************')
    
    vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
    print('vae_post_quant_conv_wrapper.model start ****************')
    vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
        # torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False
        torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False
    )
    print('vae_post_quant_conv_wrapper.model end ****************')
    
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    pipe.vae.decoder = vae_decoder_wrapper
    pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper
        
    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("\n开始 warmup 推理...")
    start = time.time()
    output_warmup = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,  # Compiled with 512x512
        width=512,  # Compiled with 512x512
        num_frames=81,  # Reduced from 7 to lower memory usage, produces latent_frames=2: (5-1)//4+1=2
        guidance_scale=5.0,
        num_inference_steps=50,
        max_sequence_length=seqlen,  # default: 512
        generator=torch.Generator().manual_seed(SEED + 1000)  # warmup使用不同的种子
    ).frames[0]
    end = time.time()
    print('warmup time:', end-start)

    # 重置generator以确保主推理使用一致的随机种子
    generator = torch.Generator().manual_seed(SEED)

    print("\n开始主推理（使用固定种子）...")
    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,  # Compiled with 512x512
        width=512,  # Compiled with 512x512
        num_frames=81,  # Reduced from 7 to lower memory usage, produces latent_frames=2: (5-1)//4+1=2
        guidance_scale=5.0,
        num_inference_steps=50,  # default: 50
        max_sequence_length=seqlen,  # default: 512
        generator=generator  # 使用固定种子的generator
    ).frames[0]
    end = time.time()
    print('time:', end-start)
    print(f"Output shape: {output.shape}")
    print(f"Output frames: {len(output)}")

    # 保存视频
    export_to_video(output, "output.mp4", fps=24)
    print("\n视频已保存到: output.mp4")

    # 可选: 保存numpy数组用于数值对比
    # 取消下面的注释以保存numpy数组
    # npy.save("output_trainium.npy", output)
    # print("Numpy数组已保存到: output_trainium.npy")
    # print("使用以下命令对比GPU和Trainium的输出:")
    # print("  python compare_outputs.py output_gpu.npy output_trainium.npy")
