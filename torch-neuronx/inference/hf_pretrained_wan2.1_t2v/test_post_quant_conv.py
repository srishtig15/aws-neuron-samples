import os
import numpy as np

import torch
from diffusers import AutoencoderKLWan, WanPipeline

import torch_neuronx


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.1_t2v_hf_cache_dir"
# HUGGINGFACE_CACHE_DIR = "wan2.1_t2v_14b_hf_cache_dir"
post_quant_conv_model_path = f"{COMPILED_MODELS_DIR}/post_quant_conv/"

DTYPE=torch.bfloat16
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR)
pipe.transformer.eval()

def inspect_neuron_model_internals():
    """深入检查 Neuron 模型的内部结构"""
    
    print("="*60)
    print("Deep Inspection of Neuron Model Internals")
    print("="*60)
    
    model_path = os.path.join(post_quant_conv_model_path, 'model.pt')
    jit_model = torch.jit.load(model_path)
    
    # 1. 检查 _c 属性（C++ module）
    if hasattr(jit_model, '_c'):
        print("✓ Found _c attribute (C++ module)")
        cpp_module = jit_model._c
        print(f"  Type: {type(cpp_module)}")
        
        # 尝试访问参数
        try:
            # 这应该是实际参数存储的地方
            param_dict = torch._C.ParameterDict(cpp_module)
            print(f"  ParameterDict type: {type(param_dict)}")
            
            # 尝试获取参数列表
            params = list(param_dict.items())
            print(f"  Number of parameters: {len(params)}")
            
            if len(params) > 0:
                print("  ✅ Parameters found in C++ module!")
                for i, (name, param) in enumerate(params[:3]):
                    print(f"    {name}: shape={param.shape}, dtype={param.dtype}")
            else:
                print("  ⚠️ No parameters in ParameterDict")
                
        except Exception as e:
            print(f"  Cannot access ParameterDict: {e}")
            
        # 尝试获取缓冲区
        try:
            buffer_dict = torch._C.BufferDict(cpp_module) 
            buffers = list(buffer_dict.items())
            print(f"  Number of buffers: {len(buffers)}")
        except Exception as e:
            print(f"  Cannot access BufferDict: {e}")
    
    # 2. 对于 NeuronModule，参数可能在不同的地方
    print("\n=== NeuronModule Specific Checks ===")
    
    # 检查 model 属性（从 graph 中看到的）
    if hasattr(jit_model, 'model'):
        print("✓ Found 'model' attribute")
        neuron_model = jit_model.model
        print(f"  Type: {type(neuron_model)}")
        # 这是 __torch__.torch.classes.neuron.Model
    
    # 3. 尝试通过 state_dict 访问
    print("\n=== State Dict Check ===")
    try:
        state_dict = jit_model.state_dict()
        print(f"State dict size: {len(state_dict)}")
        
        if len(state_dict) > 0:
            print("✅ State dict has entries:")
            for key in list(state_dict.keys())[:5]:
                value = state_dict[key]
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print("⚠️ State dict is empty")
            
    except Exception as e:
        print(f"Cannot get state_dict: {e}")
    
    # 4. 检查文件内容
    print("\n=== File Content Analysis ===")
    
    # 加载原始文件内容
    import zipfile
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            # 列出所有文件
            file_list = zf.namelist()
            print(f"Files in archive: {file_list[:10]}")  # 前10个
            
            # 查找参数文件
            param_files = [f for f in file_list if 'weight' in f or 'bias' in f or '.pt' in f]
            print(f"Potential parameter files: {param_files[:5]}")
            
    except Exception as e:
        print(f"Cannot read as zip: {e}")

inspect_neuron_model_internals()