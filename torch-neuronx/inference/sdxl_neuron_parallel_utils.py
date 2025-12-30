# SDXL UNet Tensor Parallel Utilities
# Based on PixArt Sigma's neuron_parallel_utils.py
# Shards Attention and FeedForward layers for tensor parallelism

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads, pad_model
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils


def get_sharded_data(data, dim):
    """Shard tensor data along specified dimension based on TP rank"""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    shard_size = data.shape[dim] // tp_size

    if dim == 0:
        return data[shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    else:
        raise ValueError(f"Unsupported dim: {dim}")


def shard_sdxl_attention(tp_degree: int, attn: Attention) -> Attention:
    """
    Shard SDXL Attention module for tensor parallelism.

    - to_q, to_k, to_v: ColumnParallelLinear (shard output features)
    - to_out[0]: RowParallelLinear (shard input features)

    Args:
        tp_degree: Tensor parallel degree
        attn: Attention module to shard

    Returns:
        Sharded Attention module
    """
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0, f"inner_dim {orig_inner_dim} not divisible by heads {attn.heads}"

    orig_num_heads = attn.heads

    # Calculate padded heads for even distribution
    total_padded_heads = attn.heads + get_number_of_extra_heads(attn.heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim

    assert attn.to_q.out_features == attn.to_k.out_features == attn.to_v.out_features

    # Shard to_q
    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        orig_q.in_features,
        orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=False
    )
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if orig_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # Shard to_k
    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        orig_k.in_features,
        orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=False
    )
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # Shard to_v
    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        orig_v.in_features,
        orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=False
    )
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard to_out[0] (the Linear layer, not the Dropout)
    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        orig_out.in_features,
        orig_out.out_features,
        bias=(orig_out.bias is not None),
        input_is_parallel=True
    )
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if orig_out.bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()  # Bias is not sharded for RowParallel
    del orig_out

    # Pad model for even distribution across TP ranks
    pad_model(attn, tp_degree, orig_num_heads, wrapped_classes=(Attention,))

    return attn


def shard_sdxl_feedforward(ff: FeedForward) -> FeedForward:
    """
    Shard SDXL FeedForward module for tensor parallelism.

    SDXL FeedForward structure:
    - net[0]: GEGLU with proj Linear (in -> 4*in*2 for gating)
    - net[1]: Dropout
    - net[2]: Linear (4*in -> out)

    Sharding:
    - net[0].proj: ColumnParallelLinear (shard output features)
    - net[2]: RowParallelLinear (shard input features)

    Args:
        ff: FeedForward module to shard

    Returns:
        Sharded FeedForward module
    """
    # Shard GEGLU projection (net[0].proj)
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        orig_proj.in_features,
        orig_proj.out_features,
        bias=(orig_proj.bias is not None),
        gather_output=False
    )
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if orig_proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del orig_proj

    # Shard output linear (net[2])
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        input_is_parallel=True
    )
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if orig_linear.bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()  # Bias is not sharded for RowParallel
    del orig_linear

    return ff


def shard_unet_attention_layers(unet, tp_degree: int):
    """
    Shard all Attention and FeedForward layers in SDXL UNet.

    Processes:
    - Down blocks (CrossAttnDownBlock2D)
    - Mid block (UNetMidBlock2DCrossAttn)
    - Up blocks (CrossAttnUpBlock2D)

    Args:
        unet: SDXL UNet2DConditionModel
        tp_degree: Tensor parallel degree

    Returns:
        UNet with sharded attention layers
    """
    sharded_count = 0

    # Process down blocks
    for block_idx, block in enumerate(unet.down_blocks):
        if hasattr(block, 'attentions') and block.attentions is not None:
            for attn_idx, attn_block in enumerate(block.attentions):
                if hasattr(attn_block, 'transformer_blocks'):
                    for tb_idx, tb in enumerate(attn_block.transformer_blocks):
                        # Shard self-attention (attn1)
                        tb.attn1 = shard_sdxl_attention(tp_degree, tb.attn1)
                        # Shard cross-attention (attn2)
                        tb.attn2 = shard_sdxl_attention(tp_degree, tb.attn2)
                        # Shard feedforward
                        tb.ff = shard_sdxl_feedforward(tb.ff)
                        sharded_count += 1

    # Process mid block
    if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        for attn_idx, attn_block in enumerate(unet.mid_block.attentions):
            if hasattr(attn_block, 'transformer_blocks'):
                for tb_idx, tb in enumerate(attn_block.transformer_blocks):
                    tb.attn1 = shard_sdxl_attention(tp_degree, tb.attn1)
                    tb.attn2 = shard_sdxl_attention(tp_degree, tb.attn2)
                    tb.ff = shard_sdxl_feedforward(tb.ff)
                    sharded_count += 1

    # Process up blocks
    for block_idx, block in enumerate(unet.up_blocks):
        if hasattr(block, 'attentions') and block.attentions is not None:
            for attn_idx, attn_block in enumerate(block.attentions):
                if hasattr(attn_block, 'transformer_blocks'):
                    for tb_idx, tb in enumerate(attn_block.transformer_blocks):
                        tb.attn1 = shard_sdxl_attention(tp_degree, tb.attn1)
                        tb.attn2 = shard_sdxl_attention(tp_degree, tb.attn2)
                        tb.ff = shard_sdxl_feedforward(tb.ff)
                        sharded_count += 1

    print(f"Sharded {sharded_count} transformer blocks for TP={tp_degree}")
    return unet
