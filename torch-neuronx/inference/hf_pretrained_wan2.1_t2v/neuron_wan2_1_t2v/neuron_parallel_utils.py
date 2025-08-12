from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF
from transformers.models.umt5.modeling_umt5 import UMT5Attention, UMT5LayerFF
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads, pad_model
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
import torch
from torch import nn

def get_sharded_data(data, dim):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

def shard_t5_self_attention(tp_degree: int, selfAttention: T5Attention):
    orig_inner_dim = selfAttention.q.out_features
    dim_head = orig_inner_dim // selfAttention.n_heads
    original_nheads = selfAttention.n_heads
    selfAttention.n_heads = selfAttention.n_heads // tp_degree
    selfAttention.inner_dim = dim_head * selfAttention.n_heads
    orig_q = selfAttention.q
    selfAttention.q = ColumnParallelLinear(
        selfAttention.q.in_features,
        selfAttention.q.out_features,
        bias=False, 
        gather_output=False)
    selfAttention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    del(orig_q)
    orig_k = selfAttention.k
    selfAttention.k = ColumnParallelLinear(
        selfAttention.k.in_features, 
        selfAttention.k.out_features, 
        bias=(selfAttention.k.bias is not None),
        gather_output=False)
    selfAttention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    del(orig_k)
    orig_v = selfAttention.v
    selfAttention.v = ColumnParallelLinear(
        selfAttention.v.in_features, 
        selfAttention.v.out_features, 
        bias=(selfAttention.v.bias is not None),
        gather_output=False)
    selfAttention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    del(orig_v)
    orig_out = selfAttention.o
    selfAttention.o = RowParallelLinear(
        selfAttention.o.in_features,
        selfAttention.o.out_features,
        bias=(selfAttention.o.bias is not None),
        input_is_parallel=True)
    selfAttention.o.weight.data = get_sharded_data(orig_out.weight.data, 1)
    del(orig_out)
    return selfAttention

def shard_t5_ff(ff: T5LayerFF):
    orig_wi_0 = ff.DenseReluDense.wi_0
    ff.DenseReluDense.wi_0 = ColumnParallelLinear(
        orig_wi_0.in_features,
        orig_wi_0.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_0.weight.data = get_sharded_data(orig_wi_0.weight.data, 0)
    orig_wi_1 = ff.DenseReluDense.wi_1
    ff.DenseReluDense.wi_1 = ColumnParallelLinear(
        orig_wi_1.in_features,
        orig_wi_1.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_1.weight.data = get_sharded_data(orig_wi_1.weight.data, 0)
    orig_wo = ff.DenseReluDense.wo
    ff.DenseReluDense.wo = RowParallelLinear(
        orig_wo.in_features,
        orig_wo.out_features,
        bias=False,
        input_is_parallel=True)
    ff.DenseReluDense.wo.weight.data = get_sharded_data(orig_wo.weight.data, 1)
    ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
    return ff

def shard_umt5_self_attention(tp_degree: int, selfAttention: UMT5Attention):
    orig_inner_dim = selfAttention.q.out_features
    dim_head = orig_inner_dim // selfAttention.n_heads
    original_nheads = selfAttention.n_heads
    selfAttention.n_heads = selfAttention.n_heads // tp_degree
    selfAttention.inner_dim = dim_head * selfAttention.n_heads
    orig_q = selfAttention.q
    selfAttention.q = ColumnParallelLinear(
        selfAttention.q.in_features,
        selfAttention.q.out_features,
        bias=False, 
        gather_output=False)
    selfAttention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    del(orig_q)
    orig_k = selfAttention.k
    selfAttention.k = ColumnParallelLinear(
        selfAttention.k.in_features, 
        selfAttention.k.out_features, 
        bias=(selfAttention.k.bias is not None),
        gather_output=False)
    selfAttention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    del(orig_k)
    orig_v = selfAttention.v
    selfAttention.v = ColumnParallelLinear(
        selfAttention.v.in_features, 
        selfAttention.v.out_features, 
        bias=(selfAttention.v.bias is not None),
        gather_output=False)
    selfAttention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    del(orig_v)
    orig_out = selfAttention.o
    selfAttention.o = RowParallelLinear(
        selfAttention.o.in_features,
        selfAttention.o.out_features,
        bias=(selfAttention.o.bias is not None),
        input_is_parallel=True)
    selfAttention.o.weight.data = get_sharded_data(orig_out.weight.data, 1)
    del(orig_out)
    return selfAttention

def shard_umt5_ff(ff: UMT5LayerFF):
    orig_wi_0 = ff.DenseReluDense.wi_0
    ff.DenseReluDense.wi_0 = ColumnParallelLinear(
        orig_wi_0.in_features,
        orig_wi_0.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_0.weight.data = get_sharded_data(orig_wi_0.weight.data, 0)
    orig_wi_1 = ff.DenseReluDense.wi_1
    ff.DenseReluDense.wi_1 = ColumnParallelLinear(
        orig_wi_1.in_features,
        orig_wi_1.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_1.weight.data = get_sharded_data(orig_wi_1.weight.data, 0)
    orig_wo = ff.DenseReluDense.wo
    ff.DenseReluDense.wo = RowParallelLinear(
        orig_wo.in_features,
        orig_wo.out_features,
        bias=False,
        input_is_parallel=True)
    ff.DenseReluDense.wo.weight.data = get_sharded_data(orig_wo.weight.data, 1)
    ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
    return ff

def shard_transformer_attn(tp_degree: int, attn: Attention):
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0
    orig_num_heads = attn.heads
    total_padded_heads = attn.heads + get_number_of_extra_heads(attn.heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim
    assert attn.to_q.out_features == attn.to_k.out_features and attn.to_q.out_features == attn.to_v.out_features

    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del(orig_q)

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del(orig_k)

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del(orig_v)

    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        attn.to_out[0].in_features,
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True)
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_out[0].bias is not None: 
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del(orig_out)
    pad_model(attn, tp_degree, orig_num_heads, wrapped_classes=(Attention,))
    return attn


def shard_transformer_feedforward(ff: FeedForward) -> FeedForward:
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        ff.net[0].proj.in_features,
        ff.net[0].proj.out_features,
        bias=(ff.net[0].proj.bias is not None),
        gather_output=False)
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if ff.net[0].proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del(orig_proj)
    
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        ff.net[2].in_features,
        ff.net[2].out_features,
        bias=(ff.net[2].bias is not None),
        input_is_parallel=True)
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if ff.net[2].bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    del(orig_linear)
    return ff

# 新增：卷积层分片函数
def shard_conv2d(conv_layer, tp_degree, dim=0):
    """
    分片 Conv2d 层
    dim=0: 沿输出通道分片 (out_channels)
    dim=1: 沿输入通道分片 (in_channels)
    """
    if tp_degree == 1:
        return conv_layer
    
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    
    if dim == 0:  # 沿输出通道分片
        out_channels_per_rank = conv_layer.out_channels // tp_degree
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=out_channels_per_rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=(conv_layer.bias is not None),
            padding_mode=conv_layer.padding_mode
        )
        
        # 分片权重
        start_idx = tp_rank * out_channels_per_rank
        end_idx = start_idx + out_channels_per_rank
        new_conv.weight.data = conv_layer.weight.data[start_idx:end_idx].clone()
        
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[start_idx:end_idx].clone()
            
    elif dim == 1:  # 沿输入通道分片
        in_channels_per_rank = conv_layer.in_channels // tp_degree
        new_conv = nn.Conv2d(
            in_channels=in_channels_per_rank,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups // tp_degree if conv_layer.groups > 1 else 1,
            bias=(conv_layer.bias is not None),
            padding_mode=conv_layer.padding_mode
        )
        
        # 分片权重
        start_idx = tp_rank * in_channels_per_rank
        end_idx = start_idx + in_channels_per_rank
        new_conv.weight.data = conv_layer.weight.data[:, start_idx:end_idx].clone()
        
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data.clone()
    
    return new_conv

def shard_conv3d(conv_layer, tp_degree, dim=0):
    """
    分片 Conv3d 层
    dim=0: 沿输出通道分片 (out_channels)
    dim=1: 沿输入通道分片 (in_channels)
    """
    if tp_degree == 1:
        return conv_layer
    
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    
    if dim == 0:  # 沿输出通道分片
        out_channels_per_rank = conv_layer.out_channels // tp_degree
        new_conv = nn.Conv3d(
            in_channels=conv_layer.in_channels,
            out_channels=out_channels_per_rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=(conv_layer.bias is not None),
            padding_mode=conv_layer.padding_mode
        )
        
        # 分片权重
        start_idx = tp_rank * out_channels_per_rank
        end_idx = start_idx + out_channels_per_rank
        new_conv.weight.data = conv_layer.weight.data[start_idx:end_idx].clone()
        
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[start_idx:end_idx].clone()
            
    elif dim == 1:  # 沿输入通道分片
        in_channels_per_rank = conv_layer.in_channels // tp_degree
        new_conv = nn.Conv3d(
            in_channels=in_channels_per_rank,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups // tp_degree if conv_layer.groups > 1 else 1,
            bias=(conv_layer.bias is not None),
            padding_mode=conv_layer.padding_mode
        )
        
        # 分片权重
        start_idx = tp_rank * in_channels_per_rank
        end_idx = start_idx + in_channels_per_rank
        new_conv.weight.data = conv_layer.weight.data[:, start_idx:end_idx].clone()
        
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data.clone()
    
    return new_conv

def shard_linear(linear_layer, tp_degree, dim=0):
    """
    分片 Linear 层
    dim=0: 沿输出维度分片 (out_features)
    dim=1: 沿输入维度分片 (in_features)
    """
    if tp_degree == 1:
        return linear_layer
    
    if dim == 0:  # 沿输出维度分片
        return ColumnParallelLinear(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=(linear_layer.bias is not None),
            gather_output=False
        )
    elif dim == 1:  # 沿输入维度分片
        return RowParallelLinear(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=(linear_layer.bias is not None),
            input_is_parallel=True
        )