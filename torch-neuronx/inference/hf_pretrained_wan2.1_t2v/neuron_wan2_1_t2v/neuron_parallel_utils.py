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

def shard_transformer3d_attn(tp_degree: int, attn: Attention):
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

    # 分片 to_q
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

    # 分片 to_k
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

    # 分片 to_v
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

    # 处理 norm_q 和 norm_k（如果存在）
    # 这些是 RMSNorm 层，它们对每个头的特征进行归一化
    # 分片后，每个设备处理部分头，但归一化仍然是独立的
    if hasattr(attn, 'norm_q') and attn.norm_q is not None:
        # norm_q 应该调整为新的内部维度
        # RMSNorm 通常没有可学习参数，或者参数是按特征维度的
        # 我们需要根据实际的 norm 类型来处理
        if hasattr(attn.norm_q, 'weight') and attn.norm_q.weight is not None:
            # 如果有权重，需要分片
            orig_norm_q_weight = attn.norm_q.weight.data
            attn.norm_q.weight.data = get_sharded_data(orig_norm_q_weight, 0)
            attn.norm_q.normalized_shape = (new_inner_dim,)
    
    if hasattr(attn, 'norm_k') and attn.norm_k is not None:
        if hasattr(attn.norm_k, 'weight') and attn.norm_k.weight is not None:
            orig_norm_k_weight = attn.norm_k.weight.data
            attn.norm_k.weight.data = get_sharded_data(orig_norm_k_weight, 0)
            attn.norm_k.normalized_shape = (new_inner_dim,)

    # 处理额外的 I2V 相关层（如果存在）
    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        orig_add_k = attn.add_k_proj
        attn.add_k_proj = ColumnParallelLinear(
            orig_add_k.in_features,
            orig_add_k.out_features,
            bias=(orig_add_k.bias is not None),
            gather_output=False)
        attn.add_k_proj.weight.data = get_sharded_data(orig_add_k.weight.data, 0)
        if orig_add_k.bias is not None:
            attn.add_k_proj.bias.data = get_sharded_data(orig_add_k.bias.data, 0)
        del(orig_add_k)
    
    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        orig_add_v = attn.add_v_proj
        attn.add_v_proj = ColumnParallelLinear(
            orig_add_v.in_features,
            orig_add_v.out_features,
            bias=(orig_add_v.bias is not None),
            gather_output=False)
        attn.add_v_proj.weight.data = get_sharded_data(orig_add_v.weight.data, 0)
        if orig_add_v.bias is not None:
            attn.add_v_proj.bias.data = get_sharded_data(orig_add_v.bias.data, 0)
        del(orig_add_v)
    
    # 处理 norm_added_k（如果存在）
    if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
        if hasattr(attn.norm_added_k, 'weight') and attn.norm_added_k.weight is not None:
            orig_norm_added_k_weight = attn.norm_added_k.weight.data
            attn.norm_added_k.weight.data = get_sharded_data(orig_norm_added_k_weight, 0)
            attn.norm_added_k.normalized_shape = (new_inner_dim,)

    # 分片 to_out
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