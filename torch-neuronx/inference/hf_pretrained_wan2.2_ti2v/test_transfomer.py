import os
import numpy as np

import torch
from diffusers import AutoencoderKLWan, WanPipeline

import torch_neuronx
import neuronx_distributed
from neuron_wan2_2_ti2v.neuron_commons import InferenceTransformerWrapper


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.2_ti2v_hf_cache_dir"
transformer_model_path = f"{COMPILED_MODELS_DIR}/transformer"

DTYPE=torch.bfloat16
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR)
pipe.transformer.eval()


# def compare_model_weights():
#     """尝试比较原始模型和 JIT 模型的权重"""
    
#     # 原始模型
#     original_model = pipe.transformer
    
#     # JIT 模型  
#     jit_model = torch.jit.load(os.path.join(transformer_model_path, 'model.pt'))
    
#     # 尝试获取JIT模型的信息
#     print("\nJIT Model Info:")
#     print(f"JIT model type: {type(jit_model)}")
    
#     # 查看模型的graph
#     if hasattr(jit_model, 'graph'):
#         graph_str = str(jit_model.graph)
#         # 检查是否有类型转换
#         if 'aten::to' in graph_str or 'cast' in graph_str:
#             print("Found type casting operations in graph")
#         if 'bfloat16' in graph_str or 'float16' in graph_str:
#             print("Found half precision operations in graph")
    
#     # 尝试获取 state_dict
#     try:
#         # 原始模型的权重
#         original_state = original_model.state_dict()
#         print(f"Original model has {len(original_state)} parameters")
        
#         # 打印几个关键参数的统计
#         for i, (name, param) in enumerate(list(original_state.items())):  # [:100]
#             print(f"  {name}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
            
#     except Exception as e:
#         print(f"Cannot get original state_dict: {e}")
        
#     # JIT 模型可能无法获取 state_dict，但可以尝试
#     try:
#         jit_state = jit_model.state_dict()
#         print(f"\nJIT model has {len(jit_state)} parameters")
        
#         for i, (name, param) in enumerate(list(jit_state.items())):  # [:100]
#             print(f"  {name}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
#     except:
#         print("\nJIT model doesn't have accessible state_dict")
        
#         # 尝试通过 named_parameters
#         try:
#             params = list(jit_model.named_parameters())
#             print(f"\nJIT model has {len(params)} named parameters")
#         except:
#             print("Cannot access JIT model parameters")

# compare_model_weights()


batch_size = 1
in_channels = 48
frames = 4
height = 32
width = 32
hidden_size = 4096
max_sequence_length = 512

hidden_states_1b = torch.randn([batch_size, in_channels, frames, height, width], dtype=DTYPE)
timestep_1b = torch.tensor([999], dtype=torch.int64)
# timestep_1b = torch.tensor([500], dtype=torch.int64)
# timestep_1b = torch.tensor([0], dtype=torch.int64)
encoder_hidden_states_1b = torch.randn([batch_size, max_sequence_length, hidden_size], dtype=DTYPE)

with torch.no_grad():
    output_cpu = pipe.transformer(hidden_states_1b.clone(), timestep_1b.clone(), encoder_hidden_states_1b.clone(), return_dict=False)[0]
    print('output_cpu:', output_cpu.shape, output_cpu.dtype, output_cpu)  # , output_cpu


transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
print('transformer_wrapper.transformer start ****************')
# transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(
#     transformer_model_path
# )
# 加载模型
jit_model = torch.jit.load(os.path.join(transformer_model_path, 'model.pt'))
# # 关键步骤：将权重移动到 NeuronCore
# torch_neuronx.move_trace_to_device(jit_model, 0)  # 0 是设备 ID
# transformer_wrapper.transformer = jit_model
transformer_wrapper.transformer = torch_neuronx.DataParallel( 
    jit_model, [0, 1, 2, 3], False  # Use for trn2
    # jit_model, [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
)
print('transformer_wrapper.transformer end ****************')

output_neuron = transformer_wrapper(hidden_states_1b.clone(), timestep_1b.clone(), encoder_hidden_states_1b.clone())[0]
print('output_neuron:', output_neuron.shape, output_neuron.dtype, output_neuron)  # , output_neuron

diff = (output_cpu - output_neuron).abs().max()

# 建议添加更详细的数值比较
print(f"CPU output - mean: {output_cpu.mean():.6f}, std: {output_cpu.std():.6f}")
print(f"Neuron output - mean: {output_neuron.mean():.6f}, std: {output_neuron.std():.6f}")
print(f"Max absolute difference: {diff:.6f}")
print(f"Relative difference: {(diff / output_cpu.abs().max()):.6f}")

# 检查是否有NaN或Inf
print(f"CPU has NaN: {torch.isnan(output_cpu).any()}")
print(f"Neuron has NaN: {torch.isnan(output_neuron).any()}")