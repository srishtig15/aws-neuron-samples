"""
Tensor Parallelism sharding utilities for RollingForcing CausalWanModel.

Provides sharding functions for:
- UMT5 text encoder (reused from Wan2.2)
- CausalWanModel attention and FFN layers
"""
import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from transformers.models.umt5.modeling_umt5 import UMT5Attention, UMT5LayerFF


def get_sharded_data(data, dim):
    """Get sharded data for current TP rank."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    s = data.shape[dim] // tp_size
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()


# ========================
# UMT5 Text Encoder Sharding (reused from Wan2.2)
# ========================

def shard_umt5_self_attention(tp_degree: int, selfAttention: UMT5Attention):
    """Shard UMT5 self-attention for tensor parallelism."""
    orig_inner_dim = selfAttention.q.out_features
    original_nheads = selfAttention.n_heads
    dim_head = orig_inner_dim // original_nheads
    selfAttention.n_heads = original_nheads // tp_degree
    selfAttention.inner_dim = dim_head * selfAttention.n_heads

    orig_q = selfAttention.q
    selfAttention.q = ColumnParallelLinear(
        selfAttention.q.in_features, selfAttention.q.out_features,
        bias=False, gather_output=False, dtype=torch.bfloat16)
    selfAttention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    del orig_q

    orig_k = selfAttention.k
    selfAttention.k = ColumnParallelLinear(
        selfAttention.k.in_features, selfAttention.k.out_features,
        bias=(selfAttention.k.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    selfAttention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    del orig_k

    orig_v = selfAttention.v
    selfAttention.v = ColumnParallelLinear(
        selfAttention.v.in_features, selfAttention.v.out_features,
        bias=(selfAttention.v.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    selfAttention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    del orig_v

    orig_out = selfAttention.o
    selfAttention.o = RowParallelLinear(
        selfAttention.o.in_features, selfAttention.o.out_features,
        bias=(selfAttention.o.bias is not None),
        input_is_parallel=True, dtype=torch.bfloat16)
    selfAttention.o.weight.data = get_sharded_data(orig_out.weight.data, 1)
    del orig_out

    return selfAttention


def shard_umt5_ff(ff: UMT5LayerFF):
    """Shard UMT5 feed-forward for tensor parallelism."""
    orig_wi_0 = ff.DenseReluDense.wi_0
    ff.DenseReluDense.wi_0 = ColumnParallelLinear(
        orig_wi_0.in_features, orig_wi_0.out_features,
        bias=False, gather_output=False, dtype=torch.bfloat16)
    ff.DenseReluDense.wi_0.weight.data = get_sharded_data(orig_wi_0.weight.data, 0)

    orig_wi_1 = ff.DenseReluDense.wi_1
    ff.DenseReluDense.wi_1 = ColumnParallelLinear(
        orig_wi_1.in_features, orig_wi_1.out_features,
        bias=False, gather_output=False, dtype=torch.bfloat16)
    ff.DenseReluDense.wi_1.weight.data = get_sharded_data(orig_wi_1.weight.data, 0)

    orig_wo = ff.DenseReluDense.wo
    ff.DenseReluDense.wo = RowParallelLinear(
        orig_wo.in_features, orig_wo.out_features,
        bias=False, input_is_parallel=True, dtype=torch.bfloat16)
    ff.DenseReluDense.wo.weight.data = get_sharded_data(orig_wo.weight.data, 1)

    ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
    return ff


# ========================
# CausalWanModel Sharding
# ========================

def shard_causal_wan_self_attention(tp_degree, self_attn):
    """
    Shard CausalWanModel self-attention (CausalWanSelfAttention) for TP.

    CausalWanModel uses separate q, k, v, o linear layers (not to_q/to_k/to_v).
    12 heads / TP=4 = 3 heads per rank, no padding needed.

    Args:
        tp_degree: Tensor parallelism degree (4)
        self_attn: CausalWanSelfAttention module with q, k, v, o, norm_q, norm_k
    """
    # Q projection [1536 -> 1536] -> [1536 -> 384]
    orig_q = self_attn.q
    self_attn.q = ColumnParallelLinear(
        orig_q.in_features, orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    self_attn.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if orig_q.bias is not None:
        self_attn.q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # K projection [1536 -> 1536] -> [1536 -> 384]
    orig_k = self_attn.k
    self_attn.k = ColumnParallelLinear(
        orig_k.in_features, orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    self_attn.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        self_attn.k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # V projection [1536 -> 1536] -> [1536 -> 384]
    orig_v = self_attn.v
    self_attn.v = ColumnParallelLinear(
        orig_v.in_features, orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    self_attn.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        self_attn.v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # O projection [1536 -> 1536] -> [384 -> 1536]
    orig_o = self_attn.o
    self_attn.o = RowParallelLinear(
        orig_o.in_features, orig_o.out_features,
        bias=(orig_o.bias is not None),
        input_is_parallel=True, dtype=torch.bfloat16)
    self_attn.o.weight.data = get_sharded_data(orig_o.weight.data, 1)
    if orig_o.bias is not None:
        self_attn.o.bias.data = orig_o.bias.data.detach()
    del orig_o

    # Shard norm_q/norm_k weights (WanRMSNorm)
    # These are [1536] -> [384] per rank, applied via local_rms_norm
    if hasattr(self_attn, 'norm_q') and self_attn.norm_q is not None:
        if hasattr(self_attn.norm_q, 'weight') and self_attn.norm_q.weight is not None:
            self_attn.norm_q.weight.data = get_sharded_data(
                self_attn.norm_q.weight.data, 0)

    if hasattr(self_attn, 'norm_k') and self_attn.norm_k is not None:
        if hasattr(self_attn.norm_k, 'weight') and self_attn.norm_k.weight is not None:
            self_attn.norm_k.weight.data = get_sharded_data(
                self_attn.norm_k.weight.data, 0)

    return self_attn


def shard_causal_wan_cross_attention(tp_degree, cross_attn):
    """
    Shard CausalWanModel cross-attention for TP.

    Same structure as self-attention: q, k, v, o linear layers.
    Q from video, K/V from text.
    """
    # Q projection [1536 -> 1536] -> [1536 -> 384]
    orig_q = cross_attn.q
    cross_attn.q = ColumnParallelLinear(
        orig_q.in_features, orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    cross_attn.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if orig_q.bias is not None:
        cross_attn.q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # K projection [1536 -> 1536] -> [1536 -> 384]
    # (text is already projected to dim=1536 by text_embedding before cross-attn)
    orig_k = cross_attn.k
    cross_attn.k = ColumnParallelLinear(
        orig_k.in_features, orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    cross_attn.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        cross_attn.k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # V projection [1536 -> 1536] -> [1536 -> 384]
    orig_v = cross_attn.v
    cross_attn.v = ColumnParallelLinear(
        orig_v.in_features, orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    cross_attn.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        cross_attn.v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # O projection [1536 -> 1536] -> [384 -> 1536]
    orig_o = cross_attn.o
    cross_attn.o = RowParallelLinear(
        orig_o.in_features, orig_o.out_features,
        bias=(orig_o.bias is not None),
        input_is_parallel=True, dtype=torch.bfloat16)
    cross_attn.o.weight.data = get_sharded_data(orig_o.weight.data, 1)
    if orig_o.bias is not None:
        cross_attn.o.bias.data = orig_o.bias.data.detach()
    del orig_o

    # Shard norm_q/norm_k weights
    if hasattr(cross_attn, 'norm_q') and cross_attn.norm_q is not None:
        if hasattr(cross_attn.norm_q, 'weight') and cross_attn.norm_q.weight is not None:
            cross_attn.norm_q.weight.data = get_sharded_data(
                cross_attn.norm_q.weight.data, 0)

    if hasattr(cross_attn, 'norm_k') and cross_attn.norm_k is not None:
        if hasattr(cross_attn.norm_k, 'weight') and cross_attn.norm_k.weight is not None:
            cross_attn.norm_k.weight.data = get_sharded_data(
                cross_attn.norm_k.weight.data, 0)

    return cross_attn


def shard_causal_wan_ffn(tp_degree, ffn):
    """
    Shard CausalWanModel FFN for TP.

    FFN structure: ffn.0 (Linear 1536->8960), ffn.1 (GELU), ffn.2 (Linear 8960->1536)
    The Wan FFN uses a gate structure: ffn = [up_proj, gate_act, down_proj]
    or ffn.0 = Linear(1536, 2*8960) with split, ffn.2 = Linear(8960, 1536)

    CausalWanModel uses nn.Sequential:
    ffn[0] = nn.Linear(dim, ffn_dim)  # 1536 -> 8960
    ffn[1] = nn.GELU(approximate="tanh")
    ffn[2] = nn.Linear(ffn_dim, dim)  # 8960 -> 1536
    """
    # Up projection [1536 -> 8960] -> [1536 -> 2240]
    orig_up = ffn[0]
    ffn[0] = ColumnParallelLinear(
        orig_up.in_features, orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False, dtype=torch.bfloat16)
    ffn[0].weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        ffn[0].bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Down projection [8960 -> 1536] -> [2240 -> 1536]
    orig_down = ffn[2]
    ffn[2] = RowParallelLinear(
        orig_down.in_features, orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True, dtype=torch.bfloat16)
    ffn[2].weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        ffn[2].bias.data = orig_down.bias.data.detach()
    del orig_down

    return ffn
