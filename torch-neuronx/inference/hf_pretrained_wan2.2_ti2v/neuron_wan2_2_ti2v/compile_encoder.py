import os
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Use for trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Use for trn2
compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """  # For trn2,  --verbose=INFO
# compiler_flags = """ --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """  # For trn1/inf2,  --verbose=INFO
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import argparse
import torch_neuronx
from neuron_commons import attention_wrapper

torch.nn.functional.scaled_dot_product_attention = attention_wrapper

def compile_encoder(args):
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    height = args.height
    width = args.width

    batch_size = 1
    # For I2V task, encoder only processes initial image once
    # Use 2 frames to avoid compiler edge case bug with single frame causal conv
    encoder_frames = 2

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
    )

    encoder = vae.encoder
    encoder.eval()

    # Check patchify configuration
    patch_size = vae.config.patch_size if vae.config.patch_size is not None else 1
    in_channels = vae.config.in_channels  # 12 if patch_size=2, else 3
    patchified_height = height // patch_size
    patchified_width = width // patch_size

    # print('encoder:', encoder)
    print(f'\nConfiguration:')
    print(f'  patch_size: {patch_size}')
    print(f'  Original resolution: {height}x{width}')
    print(f'  After patchify: {patchified_height}x{patchified_width}, {in_channels} channels')
    print(f'  Encoder input: (batch_size={batch_size}, channels={in_channels}, frames={encoder_frames}, height={patchified_height}, width={patchified_width})')
    
    with torch.no_grad():
        # Encoder input: AFTER patchify
        # Compile with CACHE_T=2 frames for causal convolution temporal caching
        # If patch_size=2: (1, 12, 2, 256, 256)
        # If patch_size=1 or None: (1, 3, 2, 512, 512)
        encoder_input = torch.rand(
            (batch_size, in_channels, encoder_frames, patchified_height, patchified_width),
            dtype=torch.float32
        )
        print(f'encoder_input shape: {encoder_input.shape}')

        # Trace encoder WITHOUT feat_cache
        # For I2V, encoder only processes initial image once - no temporal caching needed
        print("\nTracing encoder (without feat_cache for I2V single-image encoding)...")
        compiled_encoder = torch_neuronx.trace(
            encoder,
            encoder_input,
            compiler_workdir=f"{compiler_workdir}/encoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False
        )

        # Save compiled model
        compiled_model_dir = f"{compiled_models_dir}/encoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        torch.jit.save(compiled_encoder, f"{compiled_model_dir}/model.pt")
        print(f"\nCompiled encoder saved to: {compiled_model_dir}/model.pt")

        # Compile quant_conv (separate from encoder, like post_quant_conv in decoder)
        # quant_conv does not use feat_cache, just channel conversion
        print("\n" + "="*80)
        print("Compiling quant_conv...")
        print("="*80)

        # quant_conv input: encoder output
        # Encoder output shape: (batch_size, z_dim*2, output_frames, latent_h, latent_w)
        # For 512x512 input: (1, 32, output_frames, 32, 32)
        # Note: encoder does 4x temporal compression, so 2 input frames → output_frames depends on caching behavior
        # For I2V: quant_conv will process encoder output with variable frame count
        # We compile with encoder_frames to match encoder compilation
        z_channels = vae.config.z_dim * 2  # 32 channels

        # quant_conv input has same spatial size as encoder output (32x32 for 512x512)
        quant_conv_input = torch.rand(
            (batch_size, z_channels, encoder_frames, patchified_height//8, patchified_width//8),
            dtype=torch.float32
        )

        print(f'quant_conv input shape: {quant_conv_input.shape}')
        print(f'quant_conv: {z_channels} → {z_channels} channels (1x1x1 conv)')

        compiled_quant_conv = torch_neuronx.trace(
            vae.quant_conv,
            quant_conv_input,
            compiler_workdir=f"{compiler_workdir}/quant_conv",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False
        )

        # Save compiled quant_conv
        compiled_model_dir = f"{compiled_models_dir}/quant_conv"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        torch.jit.save(compiled_quant_conv, f"{compiled_model_dir}/model.pt")
        print(f"\nCompiled quant_conv saved to: {compiled_model_dir}/model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video.", type=int, default=512)
    parser.add_argument("--width", help="height of generated video.", type=int, default=512)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_encoder(args)