import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads, pad_model
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils


class ShardedRMSNorm(nn.Module):
    """RMSNorm that works with sharded hidden dimensions."""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        # RMSNorm computation - normalize over last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.weight is not None:
            return x_normed * self.weight
        return x_normed


def get_sharded_data(data, dim):
    """Shard data across tensor parallel ranks."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()


def shard_rmsnorm(orig_norm, new_dim):
    """Create a sharded RMSNorm from an original RMSNorm."""
    eps = orig_norm.eps if hasattr(orig_norm, 'eps') else 1e-6
    elementwise_affine = hasattr(orig_norm, 'weight') and orig_norm.weight is not None

    new_norm = ShardedRMSNorm(new_dim, eps=eps, elementwise_affine=elementwise_affine)

    if elementwise_affine and orig_norm.weight is not None:
        new_norm.weight.data = get_sharded_data(orig_norm.weight.data, 0)

    return new_norm


def shard_qwen_attention(tp_degree: int, attn: Attention):
    """
    Shard QwenImage attention module for tensor parallelism.
    This handles both image attention (to_q/k/v) and text attention (add_q/k/v_proj).
    """
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0
    orig_num_heads = attn.heads
    total_padded_heads = attn.heads + get_number_of_extra_heads(attn.heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim

    # Shard image attention projections (to_q, to_k, to_v)
    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard output projection
    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        attn.to_out[0].in_features,
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True)
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_out[0].bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del orig_out

    # Shard text attention projections (add_q_proj, add_k_proj, add_v_proj)
    if hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
        orig_add_q = attn.add_q_proj
        attn.add_q_proj = ColumnParallelLinear(
            orig_add_q.in_features,
            orig_add_q.out_features,
            bias=(orig_add_q.bias is not None),
            gather_output=False)
        attn.add_q_proj.weight.data = get_sharded_data(orig_add_q.weight.data, 0)
        if orig_add_q.bias is not None:
            attn.add_q_proj.bias.data = get_sharded_data(orig_add_q.bias.data, 0)
        del orig_add_q

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
        del orig_add_k

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
        del orig_add_v

    # Shard to_add_out
    if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
        orig_add_out = attn.to_add_out
        attn.to_add_out = RowParallelLinear(
            orig_add_out.in_features,
            orig_add_out.out_features,
            bias=(orig_add_out.bias is not None),
            input_is_parallel=True)
        attn.to_add_out.weight.data = get_sharded_data(orig_add_out.weight.data, 1)
        if orig_add_out.bias is not None:
            attn.to_add_out.bias.data = orig_add_out.bias.data.detach()
        del orig_add_out

    # Note: RMSNorm layers (norm_q, norm_k, norm_added_q, norm_added_k) should NOT be sharded!
    # They operate on head_dim (128) which doesn't change with tensor parallelism.
    # The norms are applied AFTER unflatten to [batch, seq, heads, head_dim],
    # so they normalize over head_dim, not inner_dim.

    # Note: pad_model is not needed when heads are evenly divisible by tp_degree
    # For QwenImage: 24 heads / 4 = 6 heads per rank (evenly divisible)
    return attn


def shard_feedforward(ff: FeedForward) -> FeedForward:
    """Shard FeedForward module for tensor parallelism."""
    # Shard the first linear layer (GELU projection)
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        ff.net[0].proj.in_features,
        ff.net[0].proj.out_features,
        bias=(ff.net[0].proj.bias is not None),
        gather_output=False)
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if ff.net[0].proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del orig_proj

    # Shard the output linear layer
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        ff.net[2].in_features,
        ff.net[2].out_features,
        bias=(ff.net[2].bias is not None),
        input_is_parallel=True)
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if ff.net[2].bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    del orig_linear
    return ff


def shard_qwen2_attention(tp_degree: int, self_attn):
    """
    Shard Qwen2/Qwen2.5-VL self attention module (used in text encoder).

    Handles GQA (Grouped Query Attention) where num_key_value_heads < num_heads.
    For Qwen2.5-VL: num_heads=28, num_key_value_heads=4
    With tp_degree=4: num_heads_per_rank=7, num_kv_heads_per_rank=1
    """
    # Get original dimensions
    orig_q = self_attn.q_proj
    orig_k = self_attn.k_proj
    orig_v = self_attn.v_proj
    orig_o = self_attn.o_proj

    # Check if we can shard KV heads evenly
    num_kv_heads = getattr(self_attn, 'num_key_value_heads', self_attn.num_heads)
    if num_kv_heads % tp_degree != 0:
        raise ValueError(
            f"num_key_value_heads ({num_kv_heads}) must be divisible by tp_degree ({tp_degree})")

    # Update number of heads
    self_attn.num_heads = self_attn.num_heads // tp_degree
    if hasattr(self_attn, 'num_key_value_heads'):
        self_attn.num_key_value_heads = self_attn.num_key_value_heads // tp_degree

    # Shard Q projection
    self_attn.q_proj = ColumnParallelLinear(
        orig_q.in_features,
        orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=False)
    self_attn.q_proj.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if orig_q.bias is not None:
        self_attn.q_proj.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # Shard K projection (may have smaller out_features for GQA)
    self_attn.k_proj = ColumnParallelLinear(
        orig_k.in_features,
        orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=False)
    self_attn.k_proj.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        self_attn.k_proj.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # Shard V projection (may have smaller out_features for GQA)
    self_attn.v_proj = ColumnParallelLinear(
        orig_v.in_features,
        orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=False)
    self_attn.v_proj.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        self_attn.v_proj.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard O projection
    self_attn.o_proj = RowParallelLinear(
        orig_o.in_features,
        orig_o.out_features,
        bias=(orig_o.bias is not None),
        input_is_parallel=True)
    self_attn.o_proj.weight.data = get_sharded_data(orig_o.weight.data, 1)
    if orig_o.bias is not None:
        self_attn.o_proj.bias.data = orig_o.bias.data.detach()
    del orig_o

    return self_attn


def shard_vision_attention(tp_degree: int, attn):
    """
    Shard Qwen2.5-VL Vision Encoder attention module.

    Vision attention uses fused QKV projection:
    - qkv: (in_features, 3 * in_features) -> splits into Q, K, V
    - proj: output projection
    """
    orig_qkv = attn.qkv
    orig_proj = attn.proj

    # Shard fused QKV projection
    attn.qkv = ColumnParallelLinear(
        orig_qkv.in_features,
        orig_qkv.out_features,
        bias=(orig_qkv.bias is not None),
        gather_output=False)
    attn.qkv.weight.data = get_sharded_data(orig_qkv.weight.data, 0)
    if orig_qkv.bias is not None:
        attn.qkv.bias.data = get_sharded_data(orig_qkv.bias.data, 0)
    del orig_qkv

    # Shard output projection
    attn.proj = RowParallelLinear(
        orig_proj.in_features,
        orig_proj.out_features,
        bias=(orig_proj.bias is not None),
        input_is_parallel=True)
    attn.proj.weight.data = get_sharded_data(orig_proj.weight.data, 1)
    if orig_proj.bias is not None:
        attn.proj.bias.data = orig_proj.bias.data.detach()
    del orig_proj

    return attn


def shard_vision_mlp(mlp):
    """
    Shard Qwen2.5-VL Vision Encoder MLP module.

    Uses gate_proj, up_proj, down_proj like Qwen2 MLP.
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    # Shard gate projection
    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features,
        orig_gate.out_features,
        bias=(orig_gate.bias is not None),
        gather_output=False)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    # Shard up projection
    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features,
        orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Shard down projection
    mlp.down_proj = RowParallelLinear(
        orig_down.in_features,
        orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp


def shard_qwen2_mlp(mlp):
    """
    Shard Qwen2 MLP module (used in text encoder).
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    # Shard gate projection
    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features,
        orig_gate.out_features,
        bias=(orig_gate.bias is not None),
        gather_output=False)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    # Shard up projection
    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features,
        orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Shard down projection
    mlp.down_proj = RowParallelLinear(
        orig_down.in_features,
        orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp
