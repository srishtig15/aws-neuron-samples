import os
import numpy as np
import torch
import torch.nn.functional as F

def test_env_variable_impact():
    """测试环境变量对结果的影响"""
    
    DTYPE = torch.bfloat16
    
    # 准备测试输入
    test_input = torch.randn(1, 256, 256, dtype=DTYPE)
    
    # 测试 SiLU 的差异
    print("Testing SiLU implementations:")
    
    # PyTorch 原生 SiLU
    silu_native = F.silu(test_input)
    
    # 手动实现的 SiLU
    silu_manual = test_input * torch.sigmoid(test_input)
    
    diff = (silu_native - silu_manual).abs().max()
    print(f"Native vs Manual SiLU difference: {diff:.6f}")
    
    # 测试 Softmax 在不同输入范围下的表现
    print("\nTesting Softmax with different input ranges:")
    
    for scale in [0.1, 1.0, 10.0, 100.0]:
        input_scaled = test_input * scale
        softmax_out = F.softmax(input_scaled, dim=-1)
        print(f"Scale {scale}: output mean={softmax_out.mean():.6f}, std={softmax_out.std():.6f}")

test_env_variable_impact()

def analyze_silu_difference():
    """深入分析 SiLU 差异"""
    
    # 在 bfloat16 下测试
    test_values = torch.tensor(
        [-10, -1, -0.5, 0, 0.5, 1, 10], 
        dtype=torch.bfloat16
    )
    
    for x in test_values:
        # PyTorch native
        silu_native = torch.nn.functional.silu(x)
        
        # Manual computation
        sigmoid_val = torch.sigmoid(x)
        silu_manual = x * sigmoid_val
        
        # Difference
        diff = abs(silu_native - silu_manual)
        
        print(f"x={x:6.2f}: native={silu_native:8.5f}, manual={silu_manual:8.5f}, "
              f"diff={diff:8.5f}, sigmoid={sigmoid_val:8.5f}")

analyze_silu_difference()

def visualize_softmax_scaling():
    """可视化 Softmax 的尺度效应"""
    
    import matplotlib.pyplot as plt
    
    x = torch.randn(1, 100, dtype=torch.bfloat16)
    
    scales = [0.1, 1.0, 10.0, 100.0]
    
    for scale in scales:
        scaled_x = x * scale
        softmax_out = torch.softmax(scaled_x, dim=-1)
        
        # 分析输出分布
        max_prob = softmax_out.max()
        min_prob = softmax_out.min()
        entropy = -(softmax_out * torch.log(softmax_out + 1e-10)).sum()
        
        print(f"Scale {scale:5.1f}:")
        print(f"  Max prob: {max_prob:.6f}")
        print(f"  Min prob: {min_prob:.6f}")  
        print(f"  Entropy: {entropy:.6f}")
        print(f"  Effective range: {max_prob - min_prob:.6f}")

visualize_softmax_scaling()

def estimate_cumulative_error():
    """估计累积误差"""
    
    # Transformer 中的典型计算流程
    num_layers = 28  # 假设的层数
    
    # 每层的误差来源
    silu_error_per_layer = 0.015625  # 从测试得到
    softmax_error_per_layer = 0.01  # 估计值
    
    # 最坏情况：误差累积
    worst_case_error = num_layers * (silu_error_per_layer + softmax_error_per_layer)
    
    print(f"Estimated worst-case accumulated error: {worst_case_error:.4f}")
    
    # 更现实的情况：误差部分抵消
    realistic_error = np.sqrt(num_layers) * (silu_error_per_layer + softmax_error_per_layer)
    
    print(f"Estimated realistic accumulated error: {realistic_error:.4f}")

estimate_cumulative_error()