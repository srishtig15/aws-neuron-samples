import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from diffusers.models.embeddings import get_1d_rotary_pos_embed

class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

class WanRotaryPosEmbedNew(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs_cos_list = []
        freqs_sin_list = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        
        for dim in [t_dim, h_dim, w_dim]:
            # 使用 use_real=True, repeat_interleave_real=False
            # 返回的维度是 [seq_len, dim] (通过 concatenate 实现)
            freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, 
                use_real=True,
                repeat_interleave_real=False,  # concatenated 方式
                freqs_dtype=freqs_dtype
            )
            # 只取前半部分（因为后半部分是重复的）
            freqs_cos_list.append(freqs_cos[:, :dim//2])
            freqs_sin_list.append(freqs_sin[:, :dim//2])
        
        # 分别存储cos和sin, 每个都是完整的 dim 维度
        self.freqs_cos = torch.cat(freqs_cos_list, dim=1)  # [seq_len, (t_dim + h_dim + w_dim)//2]
        self.freqs_sin = torch.cat(freqs_sin_list, dim=1)  # [seq_len, (t_dim + h_dim + w_dim)//2]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        # 处理cos和sin频率
        freqs_cos = self.freqs_cos.to(hidden_states.device)
        freqs_sin = self.freqs_sin.to(hidden_states.device)
        
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        
        # 使用完整的维度（因为 concatenate 后维度是完整的 dim）
        sizes = [t_dim//2, h_dim//2, w_dim//2]
        
        freqs_cos_split = freqs_cos.split_with_sizes(sizes, dim=1)
        freqs_sin_split = freqs_sin.split_with_sizes(sizes, dim=1)

        # 构建3D频率（cos部分）
        freqs_f_cos = freqs_cos_split[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h_cos = freqs_cos_split[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w_cos = freqs_cos_split[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        
        # 构建3D频率（sin部分）
        freqs_f_sin = freqs_sin_split[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h_sin = freqs_sin_split[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w_sin = freqs_sin_split[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        
        # 合并
        freqs_cos_combined = torch.cat([freqs_f_cos, freqs_h_cos, freqs_w_cos], dim=-1)
        freqs_sin_combined = torch.cat([freqs_f_sin, freqs_h_sin, freqs_w_sin], dim=-1)
        
        # 重塑为 [1, 1, seq_len, dim]
        freqs_cos_combined = freqs_cos_combined.reshape(1, 1, ppf * pph * ppw, -1)
        freqs_sin_combined = freqs_sin_combined.reshape(1, 1, ppf * pph * ppw, -1)
        
        # 返回 tuple (cos, sin)
        return (freqs_cos_combined, freqs_sin_combined)

def apply_rotary_emb_complex(hidden_states, freqs_complex):
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs_complex).flatten(3, 4)
    return x_out.type_as(hidden_states)

def apply_rotary_emb(hidden_states, freqs):
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    
    cos, sin = freqs
    batch, heads, seq_len, head_dim = hidden_states.shape
    
    # 重组为分离实部和虚部 (matching unflatten(3, (-1, 2)))
    x = hidden_states.to(dtype).reshape(batch, heads, seq_len, head_dim // 2, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]
    
    # 复数乘法
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    
    # 重新 interleave
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    
    return out.type_as(hidden_states)

def debug_rope_implementation():
    """比较两种实现在实际使用中的差异"""
    
    # 创建一个小型测试场景
    torch.manual_seed(42)
    
    # 模拟 Wan 模型的配置
    batch_size = 1
    num_frames, height, width = 8, 16, 16
    num_channels = 2  # 16
    patch_size = (1, 1, 1)  # (1, 2, 2)
    attention_head_dim = 8  # 128
    max_seq_len = 1024
    
    # 创建输入
    hidden_states = torch.randn(batch_size, num_channels, num_frames, height, width)
    
    # 初始化两个版本的 RoPE
    rope_complex = WanRotaryPosEmbed(attention_head_dim, patch_size, max_seq_len)
    rope_real = WanRotaryPosEmbedNew(attention_head_dim, patch_size, max_seq_len)
    
    # 获取频率编码
    freqs_complex = rope_complex(hidden_states)
    freqs_real = rope_real(hidden_states)
    print('freqs_complex:', freqs_complex)
    print('freqs_real:', freqs_real)
    
    complex_real = freqs_complex.real
    complex_imag = freqs_complex.imag
    print('complex_real:', complex_real)
    print('complex_imag:', complex_imag)
    
    freqs_cos = freqs_real[0]
    freqs_sin = freqs_real[1]
    print('freqs_cos:', freqs_cos)
    print('freqs_sin:', freqs_sin)
    
    print('complex_real:', complex_real.shape, complex_real.dtype, 'complex_imag:', complex_imag.shape, complex_imag.dtype)
    print('freqs_cos:', freqs_cos[:, :, :, :attention_head_dim//2].shape, freqs_cos[:, :, :, :attention_head_dim//2].dtype, 'freqs_sin:', freqs_sin[:, :, :, :attention_head_dim//2].shape, freqs_sin[:, :, :, :attention_head_dim//2].dtype)

    # 检查 cos 的前半部分是否等于复数的实部
    print(f"\nCos first half matches complex real? {torch.allclose(freqs_cos[:, :, :, :attention_head_dim//2].type_as(complex_real), complex_real)}")
    print(f"Sin first half matches complex imag? {torch.allclose(freqs_sin[:, :, :, :attention_head_dim//2].type_as(complex_imag), complex_imag)}")
    
    # 创建测试的 Q, K
    ppf, pph, ppw = num_frames // patch_size[0], height // patch_size[1], width // patch_size[2]
    seq_len = ppf * pph * ppw
    
    test_qk = torch.randn(batch_size, 8, seq_len, attention_head_dim)  # 8 heads
    
    # 应用两种 rotary embedding
    out_complex = apply_rotary_emb_complex(test_qk, freqs_complex)
    out_real = apply_rotary_emb(test_qk, freqs_real)
    
    print('out_complex:', out_complex)
    print('out_real:', out_real)
    
    # 比较
    diff = (out_complex - out_real).abs()
    print(f"QK rotation difference - Max: {diff.max():.6e}, Mean: {diff.mean():.6e}")
    
    # 检查特定位置的值
    if diff.max() > 1e-5:
        # 找出差异最大的位置
        max_diff_idx = diff.argmax()
        idx = np.unravel_index(max_diff_idx.cpu(), diff.shape)
        
        print(f"\n最大差异位置: {idx}")
        print(f"Complex: {out_complex[idx]:.6f}")
        print(f"Real: {out_real[idx]:.6f}")
        
        # 检查该位置的频率值
        b, h, s, d = idx
        if isinstance(freqs_complex, torch.Tensor):
            print(f"Freq complex at position: {freqs_complex[0, 0, s, d//2]}")
        print(f"Freq cos at position: {freqs_real[0][0, 0, s, d]:.6f}")
        print(f"Freq sin at position: {freqs_real[1][0, 0, s, d]:.6f}")

debug_rope_implementation()

# def test_rope_equivalence():
#     """
#     The code defines functions to test equivalence between different positional embedding
#     implementations and to debug the format of concatenated positional embeddings.
#     """
#     # 设置参数
#     batch_size = 1
#     num_heads = 2
#     seq_len = 2
#     head_dim = 8
    
#     # 创建测试输入
#     torch.manual_seed(42)
#     hidden_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
#     # 2. 获取复数形式的频率
#     freqs_complex = get_1d_rotary_pos_embed(
#         head_dim, seq_len, theta=10000.0,
#         use_real=False, 
#         repeat_interleave_real=False,
#         freqs_dtype=torch.float64
#     )
#     freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
#     print('freqs_complex:', freqs_complex)
    
#     # 3. 获取实数形式的频率
#     freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
#         head_dim, seq_len, theta=10000.0,
#         use_real=True,
#         repeat_interleave_real=False,
#         freqs_dtype=torch.float64
#     )
#     freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
#     freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
#     print('freqs_cos:', freqs_cos)
#     print('freqs_sin:', freqs_sin)
    
#     # 4. 应用两种版本
#     out_complex = apply_rotary_emb_complex(hidden_states, freqs_complex)
#     out_real = apply_rotary_emb(hidden_states, (freqs_cos, freqs_sin))
#     print('out_complex:', out_complex)
#     print('out_real:', out_real)
    
#     # 5. 比较结果
#     diff = (out_complex - out_real).abs()
#     print(f"Max difference: {diff.max().item():.6e}")
#     print(f"Mean difference: {diff.mean().item():.6e}")
#     print(f"Relative error: {(diff / (out_complex.abs() + 1e-8)).mean().item():.6e}")
    
#     # 如果差异较大，检查具体位置
#     if diff.max() > 1e-5:
#         print("\n检查差异较大的位置:")
#         idx = diff.argmax()
#         idx_tuple = np.unravel_index(idx.cpu(), diff.shape)
#         print(f"Position: {idx_tuple}")
#         print(f"Complex output: {out_complex[idx_tuple].item():.6f}")
#         print(f"Real output: {out_real[idx_tuple].item():.6f}")
        
#     return out_complex, out_real

# # 运行测试
# test_rope_equivalence()

# def debug_rope_format():
#     """理解 concatenated 形式的具体格式"""
#     import torch
    
#     # 创建一个简单的测试案例
#     head_dim = 8  # 使用小的维度便于观察
#     seq_len = 2
    
#     # 1. 获取复数形式
    
#     freqs_complex = get_1d_rotary_pos_embed(
#         head_dim, seq_len, theta=10000.0,
#         use_real=False, 
#         repeat_interleave_real=False,
#         freqs_dtype=torch.float64
#     )
#     print(f"Complex freqs shape: {freqs_complex.shape}")  # 应该是 [seq_len, head_dim//2]
#     print(f"Complex freqs:\n{freqs_complex}")
    
#     # 2. 获取实数形式
#     freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
#         head_dim, seq_len, theta=10000.0,
#         use_real=True,
#         repeat_interleave_real=False,
#         freqs_dtype=torch.float64
#     )
#     print(f"\nCos shape: {freqs_cos.shape}")  # 应该是 [seq_len, head_dim]
#     print(f"Sin shape: {freqs_sin.shape}")
#     print(f"Cos:\n{freqs_cos}")
#     print(f"Sin:\n{freqs_sin}")
    
#     # 3. 手动验证关系
#     # 对于 concatenated 形式，cos 和 sin 应该是：
#     # cos = [cos(θ₀), cos(θ₁), ..., cos(θ_{d/2-1}), cos(θ₀), cos(θ₁), ..., cos(θ_{d/2-1})]
#     # sin = [sin(θ₀), sin(θ₁), ..., sin(θ_{d/2-1}), sin(θ₀), sin(θ₁), ..., sin(θ_{d/2-1})]
    
#     # 验证复数形式的实部和虚部
#     complex_real = freqs_complex.real
#     complex_imag = freqs_complex.imag
    
#     print(f"\nComplex real part:\n{complex_real}")
#     print(f"Complex imag part:\n{complex_imag}")
    
#     # 检查 cos 的前半部分是否等于复数的实部
#     print(f"\nCos first half matches complex real? {torch.allclose(freqs_cos[:, :head_dim//2].type_as(complex_real), complex_real)}")
#     print(f"Sin first half matches complex imag? {torch.allclose(freqs_sin[:, :head_dim//2].type_as(complex_imag), complex_imag)}")

# debug_rope_format()