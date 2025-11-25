import os
# os.environ["NEURON_INTERNAL_USE_VANILLA_TORCH_XLA"] = "1"  # RuntimeError: Unknown custom-call API version enum value: 0 (API_VERSION_UNSPECIFIED) https://github.com/aws-neuron/aws-neuron-sdk/issues/789
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2,  --verbose=INFO
# compiler_flags = """ --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2. optlevel=1 creates smaller subgraphs,  --verbose=INFO
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import argparse
import torch_neuronx
from diffusers.models.autoencoders.vae import Decoder
from neuron_commons import attention_wrapper, f32Wrapper

torch.nn.functional.scaled_dot_product_attention =  attention_wrapper

def upcast_norms_to_f32(decoder: Decoder):
    for upblock in decoder.up_blocks:
        for resnet in upblock.resnets:
            orig_resnet_norm1 = resnet.norm1
            orig_resnet_norm2 = resnet.norm2
            resnet.norm1 = f32Wrapper(orig_resnet_norm1)
            resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    for attn in decoder.mid_block.attentions:
        orig_norm = attn.norm
        attn.norm = f32Wrapper(orig_norm)
    for resnet in decoder.mid_block.resnets:
        orig_resnet_norm1 = resnet.norm1
        orig_resnet_norm2 = resnet.norm2
        resnet.norm1 = f32Wrapper(orig_resnet_norm1)
        resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    orig_norm_out = decoder.norm_out
    decoder.norm_out = f32Wrapper(orig_norm_out)

def compile_decoder(args):
    # Must match transformer output: height//16 because of VAE (//8) + patch_embedding (//2)
    latent_height = args.height//16
    latent_width = args.width//16
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    batch_size = 1
    decoder_frames = 2  # Decoder needs CACHE_T=2 frames (will pad 1-frame inputs at runtime)
    latent_frames = (args.num_frames - 1) // 4 + 1  # post_quant_conv processes full latents
    print(f"num_frames={args.num_frames} -> latent_frames={latent_frames}")
    in_channels = 48
    
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.2_ti2v_hf_cache_dir")
    
    decoder: Decoder = vae.decoder
    decoder.eval()
    # print('decoder:', decoder)

    # upcast_norms_to_f32(decoder)  # TODO maybe we don't need to call upcast_norms_to_f32
        
    with torch.no_grad():
        # Decoder input: always 2 frames (CACHE_T=2)
        decoder_input = torch.rand((batch_size, in_channels, decoder_frames, latent_height, latent_width), dtype=torch.float32)

        # 根据analyze_decoder_full.py的分析结果，创建完整的feat_cache
        feat_cache = [
            torch.rand((batch_size, 48, 2, latent_height, latent_width), dtype=torch.float32),  # 0: conv_in
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 1: mid_block.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 2: mid_block.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 3: mid_block.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 4: mid_block.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 5: up_blocks.0.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 6: up_blocks.0.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 7: up_blocks.0.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 8: up_blocks.0.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 9: up_blocks.0.resnets.2.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 10: up_blocks.0.resnets.2.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 11: up_blocks.0.upsampler.time_conv
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 12: up_blocks.1.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 13: up_blocks.1.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 14: up_blocks.1.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 15: up_blocks.1.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 16: up_blocks.1.resnets.2.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 17: up_blocks.1.resnets.2.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 18: up_blocks.1.upsampler.time_conv
            torch.rand((batch_size, 1024, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 19: up_blocks.2.resnets.0.conv1
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 20: up_blocks.2.resnets.0.conv2
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 21: up_blocks.2.resnets.0.conv_shortcut
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 22: up_blocks.2.resnets.1.conv1
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 23: up_blocks.2.resnets.1.conv2
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 24: up_blocks.2.resnets.2.conv1
            torch.rand((batch_size, 512, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 25: up_blocks.2.resnets.2.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 26: up_blocks.3.resnets.0.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 27: up_blocks.3.resnets.0.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 28: up_blocks.3.resnets.0.conv_shortcut
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 29: up_blocks.3.resnets.1.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 30: up_blocks.3.resnets.1.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 31: up_blocks.3.resnets.2.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 32: up_blocks.3.resnets.2.conv2 (dummy, not used)
            torch.rand((batch_size, 12, 2, latent_height*8, latent_width*8), dtype=torch.float32),   # 33: conv_out (dummy, not used)
        ]

        # Trace decoder normally with feat_cache
        # Will load without DataParallel at runtime to support list arguments
        compiled_decoder = torch_neuronx.trace(
            decoder,
            (decoder_input, feat_cache),
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
        
        compiled_model_dir = f"{compiled_models_dir}/decoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        torch.jit.save(compiled_decoder, f"{compiled_model_dir}/model.pt")

        # post_quant_conv input: full latent_frames (e.g., 4 for num_frames=15)
        post_quant_conv_input = torch.rand((batch_size, in_channels, latent_frames, latent_height, latent_width), dtype=torch.float32)

        compiled_post_quant_conv = torch_neuronx.trace(
            vae.post_quant_conv,
            post_quant_conv_input,
            compiler_workdir=f"{compiler_workdir}/post_quant_conv",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
        
        compiled_model_dir = f"{compiled_models_dir}/post_quant_conv"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)     
        torch.jit.save(compiled_post_quant_conv, f"{compiled_model_dir}/model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video.", type=int, default=512)
    parser.add_argument("--width", help="width of generated video.", type=int, default=512)
    parser.add_argument("--num_frames", help="number of frames in generated video.", type=int, default=81)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_decoder(args)