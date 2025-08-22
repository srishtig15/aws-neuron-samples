import torch
import torch.nn as nn
import torch_neuronx
from diffusers.models.normalization import FP32LayerNorm
from neuron_wan2_1_t2v.neuron_commons import f32Wrapper


def upcast_norms_to_f32(model):
    model.norm = f32Wrapper(model.norm)

# 测试 FP32LayerNorm 在 Neuron 上的行为
def test_fp32_layernorm_on_neuron():
    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = FP32LayerNorm(768)
        
        def forward(self, x):
            return self.norm(x)
    
    model = TestModel()
    x = torch.randn(1, 100, 768, dtype=torch.bfloat16)
    
    # 测试原始模型
    output_cpu = model(x)
    print(f"CPU output dtype: {output_cpu.dtype}")  # 应该是 BF16
    
    # 编译到 Neuron
    model_neuron = torch_neuronx.trace(
        model,
        x,
        compiler_args="--model-type=transformer"
    )
    
    # 测试 Neuron 模型
    output_neuron = model_neuron(x)
    print(f"Neuron output dtype: {output_neuron.dtype}")
    
    # 比较精度
    diff = (output_cpu - output_neuron).abs().max()
    print(f"Max difference: {diff}")
    
    # Upcast norms to fp32
    upcast_norms_to_f32(model)
    
    # 编译到 Neuron
    model_neuron2 = torch_neuronx.trace(
        model,
        x,
        compiler_args="--model-type=transformer"
    )
    
    # 测试 Neuron 模型
    output_neuron2 = model_neuron2(x)
    print(f"Neuron output 2 dtype: {output_neuron2.dtype}")
    
    # 比较精度
    diff2 = (output_cpu - output_neuron2).abs().max()
    print(f"Max difference: {diff2}")
    
    # 比较编译后的图
    print(model_neuron.graph)  # 查看编译后的计算图
    print(model_neuron2.graph)
    
    return diff < 1e-3  # 检查精度是否可接受

result = test_fp32_layernorm_on_neuron()
print('result:', result)