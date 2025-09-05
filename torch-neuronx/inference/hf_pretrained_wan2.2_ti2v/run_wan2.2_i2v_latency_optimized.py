# imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_image

import numpy as np
from PIL import Image

import neuronx_distributed
import numpy as npy
import os
import time
import torch
import torch_neuronx

from neuron_wan2_2_ti2v.neuron_commons import InferenceTextEncoderWrapper
from neuron_wan2_2_ti2v.neuron_commons import InferenceTransformerWrapper
from neuron_wan2_2_ti2v.neuron_commons import SimpleWrapper

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.2_ti2v_hf_cache_dir"

def prepare_image_latents(pipe, image, num_frames, height, width, device, dtype, generator=None):
    """
    准备包含第一帧图像的latents
    """
    # 处理输入图像
    if isinstance(image, str):
        image = load_image(image)
    
    if isinstance(image, Image.Image):
        # 调整图像大小以匹配目标尺寸
        image = image.resize((width, height), Image.LANCZOS)
        image = np.array(image)
    
    # 转换为tensor并归一化
    image = torch.from_numpy(image).float() / 127.5 - 1.0
    image = image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    
    # 重要：为VAE添加时间维度 [batch, channels, num_frames=1, height, width]
    image = image.unsqueeze(2)  # [1, C, 1, H, W]
    image = image.to(device=device, dtype=dtype)
    
    # 编码图像到潜在空间
    with torch.no_grad():
        # 现在image是5维的，符合VAE的要求
        image_latents = pipe.vae.encode(image).latent_dist.sample(generator)
        
        # 应用VAE的缩放
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
        image_latents = (image_latents - latents_mean) * latents_std
    
    # 准备完整视频的latents
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial
    
    # 创建随机噪声
    shape = (1, image_latents.shape[1], num_latent_frames, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    
    # 将第一帧替换为编码的图像
    # image_latents已经是[1, C, 1, latent_H, latent_W]格式
    latents[:, :, 0:1, :, :] = image_latents.to(torch.float32)
    
    return latents

if __name__ == "__main__":    
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

    # vae_decoder_wrapper = SimpleWrapper(pipe.vae.decoder)
    # print('vae_decoder_wrapper.model start ****************')
    # vae_decoder_wrapper.model = torch_neuronx.DataParallel( 
    #     # torch.jit.load(decoder_model_path), [0, 1, 2, 3], False  # Use for trn2
    #     torch.jit.load(decoder_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    # )
    # print('vae_decoder_wrapper.model:', vae_decoder_wrapper.model)
    # print('vae_decoder_wrapper.model end ****************')
    
    # vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
    # print('vae_post_quant_conv_wrapper.model start ****************')
    # vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
    #     torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False # Use for trn2
    #     # torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    # )
    # print('vae_post_quant_conv_wrapper.model end ****************')
    
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    # pipe.vae.decoder = vae_decoder_wrapper
    # pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper
    
    height = 512
    width = 512
    num_frames = 61
    
    # 加载输入图像
    input_image = "cat.png"  # 或直接传入PIL Image对象
    input_image = load_image(input_image)

    # 准备包含第一帧的latents
    generator = torch.Generator().manual_seed(42)  # 可选，用于可重复性
    latents = prepare_image_latents(
        pipe, 
        input_image, 
        num_frames, 
        height, 
        width, 
        torch.device('cpu'), 
        dtype=torch.float32,  # latents通常使用float32
        generator=generator
    )
    print('latents:', latents.shape, latents.dtype)
        
    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # start = time.time()
    # output_warmup = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     height=height,  # default: 480
    #     width=width,  # default: 832
    #     num_frames=num_frames,  # default: 81
    #     guidance_scale=5.0,
    #     max_sequence_length=seqlen  # default: 512
    # ).frames[0]
    # end = time.time()
    # print('warmup time:', end-start)

    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,  # default: 480
        width=width,  # default: 832
        num_frames=num_frames,  # default: 81
        guidance_scale=5.0,
        num_inference_steps=50,  # default: 50
        max_sequence_length=seqlen,  # default: 512
        latents=latents,  # 传入准备好的latents
        generator=generator,
    ).frames[0]
    end = time.time()
    print('time:', end-start)
    export_to_video(output, "output.mp4", fps=15)
