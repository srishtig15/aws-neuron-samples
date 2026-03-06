"""
调试脚本：检查实际运行时的 latent shape
"""
import torch
import torch_neuronx
import neuronx_distributed

from diffusers import AutoencoderKLWan, WanPipeline
from neuron_wan2_2_ti2v.neuron_commons import InferenceTextEncoderWrapper, InferenceTransformerWrapper, SimpleWrapper, DecoderWrapper

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
DTYPE = torch.bfloat16

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Load pipeline
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR)

# Load compiled models (只加载 post_quant_conv 和 transformer)
text_encoder_model_path = f"{COMPILED_MODELS_DIR}/text_encoder"
transformer_model_path = f"{COMPILED_MODELS_DIR}/transformer"
post_quant_conv_model_path = f"{COMPILED_MODELS_DIR}/post_quant_conv/model.pt"

# Text encoder
text_encoder_wrapper = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, 512)
text_encoder_wrapper.t = neuronx_distributed.trace.parallel_model_load(text_encoder_model_path)

# Transformer
transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(transformer_model_path)

# Post quant conv
vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
    torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False
)

# Replace in pipeline
pipe.text_encoder = text_encoder_wrapper
pipe.transformer = transformer_wrapper
pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper

# Test inference
prompt = "A cat walks on the grass"
negative_prompt = "Bright tones"

print("="*80)
print("Testing with num_frames=81")
print("="*80)

# Hook to capture intermediate shapes
original_decode = pipe.vae._decode

def debug_decode(z, return_dict=True):
    print(f"\nVAE._decode called:")
    print(f"  Input z shape: {z.shape}")
    print(f"  Expected latent_frames: {(81-1)//4+1} = 21")
    print()

    # Call post_quant_conv
    print("  Calling post_quant_conv...")
    x = pipe.vae.post_quant_conv(z)
    print(f"  post_quant_conv output shape: {x.shape}")

    # Temporarily don't call decoder
    print()
    print("  Stopping here to avoid decoder errors")
    print("="*80)

    # Return dummy output
    from diffusers.models.modeling_outputs import DecoderOutput
    return DecoderOutput(sample=torch.zeros(1, 12, x.shape[2], 256, 256))

pipe.vae._decode = debug_decode

try:
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=2,  # 只跑2步用于测试
        max_sequence_length=512
    )
    print("\nInference completed")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
