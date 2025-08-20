import torch
import numpy as np
from typing import Tuple
from diffusers.models.embeddings import get_1d_rotary_pos_embed

def apply_rotary_emb_complex(hidden_states, freqs_complex):
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs_complex).flatten(3, 4)
    return x_out.type_as(hidden_states)

# def apply_rotary_emb(hidden_states: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]):
#     """
#     应用 rotary embedding（对应 repeat_interleave_real=False 的 concatenated 形式）
    
#     hidden_states: [batch, heads, seq_len, head_dim]
#     freqs: (cos, sin) 每个都是 [1, 1, seq_len, head_dim]
#     """
#     cos, sin = freqs  # 每个都是 [1, 1, seq_len, head_dim]
    
#     # 对于 concatenated 形式，hidden_states 布局为 [x1, x2, x3, ..., xd/2, xd/2+1, ..., xd]
#     # 需要将其旋转为 [-xd/2+1, -xd/2+2, ..., -xd, x1, x2, ..., xd/2]
#     d = hidden_states.shape[-1]
#     assert d % 2 == 0, f"head_dim must be even, got {d}"
    
#     # 分成两半
#     x1 = hidden_states[..., :d//2]
#     x2 = hidden_states[..., d//2:]
    
#     # 构建旋转后的向量 [-x2, x1]
#     x_rotated = torch.cat([-x2, x1], dim=-1)
    
#     # 应用旋转
#     out = (hidden_states.float() * cos + x_rotated.float() * sin).type_as(hidden_states)
    
#     return out

def apply_rotary_emb(hidden_states, freqs):
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    
    cos, sin = freqs
    batch, heads, seq_len, head_dim = hidden_states.shape
    
    # 重组为分离实部和虚部 (matching unflatten(3, (-1, 2)))
    x = hidden_states.to(dtype).reshape(batch, heads, seq_len, head_dim // 2, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]
    
    # 使用前半部分的 cos/sin（因为后半部分是重复的）
    cos_half = cos[..., :head_dim // 2]
    sin_half = sin[..., :head_dim // 2]
    
    # 复数乘法
    out_real = x_real * cos_half - x_imag * sin_half
    out_imag = x_real * sin_half + x_imag * cos_half
    
    # 重新 interleave
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    
    return out.type_as(hidden_states)

def test_rope_equivalence():
    # 设置参数
    batch_size = 2
    num_heads = 8
    seq_len = 16
    head_dim = 64
    
    # 创建测试输入
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # 2. 获取复数形式的频率
    freqs_complex = get_1d_rotary_pos_embed(
        head_dim, seq_len, theta=10000.0,
        use_real=False, 
        repeat_interleave_real=False,
        freqs_dtype=torch.float64
    )
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # 3. 获取实数形式的频率
    freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
        head_dim, seq_len, theta=10000.0,
        use_real=True,
        repeat_interleave_real=False,
        freqs_dtype=torch.float64
    )
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    
    # 4. 应用两种版本
    out_complex = apply_rotary_emb_complex(hidden_states, freqs_complex)
    out_real = apply_rotary_emb(hidden_states, (freqs_cos, freqs_sin))
    
    # 5. 比较结果
    diff = (out_complex - out_real).abs()
    print(f"Max difference: {diff.max().item():.6e}")
    print(f"Mean difference: {diff.mean().item():.6e}")
    print(f"Relative error: {(diff / (out_complex.abs() + 1e-8)).mean().item():.6e}")
    
    # 如果差异较大，检查具体位置
    if diff.max() > 1e-5:
        print("\n检查差异较大的位置:")
        idx = diff.argmax()
        idx_tuple = np.unravel_index(idx.cpu(), diff.shape)
        print(f"Position: {idx_tuple}")
        print(f"Complex output: {out_complex[idx_tuple].item():.6f}")
        print(f"Real output: {out_real[idx_tuple].item():.6f}")
        
    return out_complex, out_real

# 运行测试
test_rope_equivalence()

def debug_rope_format():
    """理解 concatenated 形式的具体格式"""
    import torch
    
    # 创建一个简单的测试案例
    head_dim = 8  # 使用小的维度便于观察
    seq_len = 2
    
    # 1. 获取复数形式
    
    freqs_complex = get_1d_rotary_pos_embed(
        head_dim, seq_len, theta=10000.0,
        use_real=False, 
        repeat_interleave_real=False,
        freqs_dtype=torch.float64
    )
    print(f"Complex freqs shape: {freqs_complex.shape}")  # 应该是 [seq_len, head_dim//2]
    print(f"Complex freqs:\n{freqs_complex}")
    
    # 2. 获取实数形式
    freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
        head_dim, seq_len, theta=10000.0,
        use_real=True,
        repeat_interleave_real=False,
        freqs_dtype=torch.float64
    )
    print(f"\nCos shape: {freqs_cos.shape}")  # 应该是 [seq_len, head_dim]
    print(f"Sin shape: {freqs_sin.shape}")
    print(f"Cos:\n{freqs_cos}")
    print(f"Sin:\n{freqs_sin}")
    
    # 3. 手动验证关系
    # 对于 concatenated 形式，cos 和 sin 应该是：
    # cos = [cos(θ₀), cos(θ₁), ..., cos(θ_{d/2-1}), cos(θ₀), cos(θ₁), ..., cos(θ_{d/2-1})]
    # sin = [sin(θ₀), sin(θ₁), ..., sin(θ_{d/2-1}), sin(θ₀), sin(θ₁), ..., sin(θ_{d/2-1})]
    
    # 验证复数形式的实部和虚部
    complex_real = freqs_complex.real
    complex_imag = freqs_complex.imag
    
    print(f"\nComplex real part:\n{complex_real}")
    print(f"Complex imag part:\n{complex_imag}")
    
    # 检查 cos 的前半部分是否等于复数的实部
    print(f"\nCos first half matches complex real? {torch.allclose(freqs_cos[:, :head_dim//2].type_as(complex_real), complex_real)}")
    print(f"Sin first half matches complex imag? {torch.allclose(freqs_sin[:, :head_dim//2].type_as(complex_imag), complex_imag)}")

debug_rope_format()