import os
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Use for trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Use for trn2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """  # For trn1/inf2
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
    encoder_frames = 1  # Compile with CACHE_T=2 frames (EncoderWrapper will pad 1-frame I2V inputs at runtime)

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="wan2.2_ti2v_hf_cache_dir"
    )

    encoder = vae.encoder
    encoder.eval()

    # Check patchify configuration
    patch_size = vae.config.patch_size if vae.config.patch_size is not None else 1
    in_channels = vae.config.in_channels  # 12 if patch_size=2, else 3
    patchified_height = height // patch_size
    patchified_width = width // patch_size

    print('encoder:', encoder)
    print(f'\nConfiguration:')
    print(f'  patch_size: {patch_size}')
    print(f'  Original resolution: {height}x{width}')
    print(f'  After patchify: {patchified_height}x{patchified_width}, {in_channels} channels')
    print(f'  Encoder input: (batch_size={batch_size}, channels={in_channels}, frames={encoder_frames}, height={patchified_height}, width={patchified_width})')
    
    with torch.no_grad():
        # Encoder input: AFTER patchify
        # Compile with 2 frames (CACHE_T=2) to avoid "Value out of range" error
        # At runtime, EncoderWrapper will pad 1-frame I2V inputs to 2 frames
        # If patch_size=2: (1, 12, 2, 256, 256)
        # If patch_size=1 or None: (1, 3, 2, 512, 512)
        encoder_input = torch.rand(
            (batch_size, in_channels, encoder_frames, patchified_height, patchified_width),
            dtype=torch.float32
        )

        # Encoder feat_cache: 24 layers (excluding conv_shortcut layers)
        # IMPORTANT: feat_cache stores INPUT shape to each conv layer
        # Spatial dimensions are AFTER patchify and downsample progressively
        # For 512x512 input with patch_size=2:
        #   - After patchify: 256x256
        #   - down_blocks.0 downsampler: 256x256 → 128x128 (spatial only)
        #   - down_blocks.1 downsampler: 128x128 → 64x64 (spatial + temporal 2x)
        #   - down_blocks.2 downsampler: 64x64 → 32x32 (spatial + temporal 2x)
        #   - down_blocks.3: no downsampler, 32x32
        feat_cache = [
            # conv_in: 12 → 160, patchified_height x patchified_width
            torch.zeros((batch_size, 12, 2, patchified_height, patchified_width), dtype=torch.float32),
            # down_blocks.0: 160 channels throughout, 256x256
            torch.zeros((batch_size, 160, 2, patchified_height, patchified_width), dtype=torch.float32),       # resnets.0.conv1 (160→160)
            torch.zeros((batch_size, 160, 2, patchified_height, patchified_width), dtype=torch.float32),       # resnets.0.conv2 (160→160)
            torch.zeros((batch_size, 160, 2, patchified_height, patchified_width), dtype=torch.float32),       # resnets.1.conv1 (160→160)
            torch.zeros((batch_size, 160, 2, patchified_height, patchified_width), dtype=torch.float32),       # resnets.1.conv2 (160→160)
            # After spatial downsample: 256x256 → 128x128
            # down_blocks.1: 160 → 320 channel increase, 128x128
            # NOTE: conv_shortcut is NOT in feat_cache (called without feat_cache argument)
            torch.zeros((batch_size, 160, 2, patchified_height//2, patchified_width//2), dtype=torch.float32),  # resnets.0.conv1 (160→320)
            torch.zeros((batch_size, 320, 2, patchified_height//2, patchified_width//2), dtype=torch.float32),  # resnets.0.conv2 (320→320)
            torch.zeros((batch_size, 320, 2, patchified_height//2, patchified_width//2), dtype=torch.float32),  # resnets.1.conv1 (320→320)
            torch.zeros((batch_size, 320, 2, patchified_height//2, patchified_width//2), dtype=torch.float32),  # resnets.1.conv2 (320→320)
            torch.zeros((batch_size, 320, 2, patchified_height//4, patchified_width//4), dtype=torch.float32),  # downsampler.time_conv (320→320) - AFTER spatial downsample to 64x64!
            # After spatial+temporal downsample: 128x128 → 64x64, temporal 2x
            # down_blocks.2: 320 → 640 channel increase, 64x64
            # NOTE: conv_shortcut is NOT in feat_cache (called without feat_cache argument)
            torch.zeros((batch_size, 320, 2, patchified_height//4, patchified_width//4), dtype=torch.float32),  # resnets.0.conv1 (320→640)
            torch.zeros((batch_size, 640, 2, patchified_height//4, patchified_width//4), dtype=torch.float32),  # resnets.0.conv2 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//4, patchified_width//4), dtype=torch.float32),  # resnets.1.conv1 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//4, patchified_width//4), dtype=torch.float32),  # resnets.1.conv2 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # downsampler.time_conv (640→640) - AFTER spatial downsample to 32x32!
            # After spatial+temporal downsample: 64x64 → 32x32, temporal 2x
            # down_blocks.3: 640 channels throughout, 32x32
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.0.conv1 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.0.conv2 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.1.conv1 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.1.conv2 (640→640)
            # mid_block: 640 channels throughout, 32x32
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.0.conv1 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.0.conv2 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.1.conv1 (640→640)
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # resnets.1.conv2 (640→640)
            # conv_out: 640 → 96, 32x32
            torch.zeros((batch_size, 640, 2, patchified_height//8, patchified_width//8), dtype=torch.float32),  # conv_out (640→96)
        ]

        print(f'feat_cache has {len(feat_cache)} elements')

        # Trace encoder with feat_cache
        # Cannot use DataParallel at runtime because it doesn't support list[Tensor] arguments
        print("\nTracing encoder...")
        compiled_encoder = torch_neuronx.trace(
            encoder,
            encoder_input,  # (encoder_input, feat_cache),
            compiler_workdir=f"{compiler_workdir}/encoder",
            compiler_args=compiler_flags.split(),
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
            compiler_args=compiler_flags.split(),
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