from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from transformers.models.umt5 import UMT5EncoderModel
import torch.jit
from torch import nn
from types import SimpleNamespace

class InferenceTextEncoderWrapper(nn.Module):
    def __init__(self, dtype, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        self.t = t
    def forward(self, text_input_ids, attention_mask=None):
        # print('self.dtype:', self.dtype)
        # print('self.device:', self.device)
        # print('self.t:', self.t)
        # print('text_input_ids:', text_input_ids)
        # print('attention_mask:', attention_mask)
        result = self.t(text_input_ids, attention_mask)  # , attention_mask
        # print('result:', type(result), result)
        # return [result['last_hidden_state'].to(self.dtype)]
        return SimpleNamespace(last_hidden_state=result['last_hidden_state'].to(self.dtype))

class InferenceTransformerWrapper(nn.Module):
    def __init__(self, transformer: WanTransformer3DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        self.cache_context = transformer.cache_context
    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, return_dict=False, **kwargs):  # encoder_attention_mask=None, added_cond_kwargs=None,
        # print('self.config:', self.config)
        # print('self.dtype:', self.dtype)
        # print('self.device:', self.device)
        # print('self.transformer:', self.transformer)
        # print('hidden_states:', hidden_states.shape, hidden_states)
        # print('timestep:', timestep)
        # print('encoder_hidden_states:', encoder_hidden_states.shape, encoder_hidden_states)
        # print('kwargs:', kwargs)
        output = self.transformer(
            hidden_states, 
            timestep,
            encoder_hidden_states
        )
        # print('output:', output.shape, output)
        return output

class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, **kwargs):
        output = self.model(x, **kwargs)
        return output

    def clear_cache(self):
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()

class DecoderWrapper(nn.Module):
    """Specialized wrapper for VAE decoder that handles TorchScript feat_cache compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Store the expected feat_cache shapes for compiled decoder
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x"""
        batch_size = x.shape[0]
        latent_height = x.shape[3]
        latent_width = x.shape[4]

        # Create dummy feat_cache with correct shapes (EXACTLY matching compile_decoder.py lines 67-100)
        # All feat_cache tensors have time dimension of 2 (CACHE_T=2)
        self.feat_cache_shapes = [
            (batch_size, 48, 2, latent_height, latent_width),  # 0: conv_in
            (batch_size, 1024, 2, latent_height, latent_width),  # 1: mid_block.resnets.0.conv1
            (batch_size, 1024, 2, latent_height, latent_width),  # 2: mid_block.resnets.0.conv2
            (batch_size, 1024, 2, latent_height, latent_width),  # 3: mid_block.resnets.1.conv1
            (batch_size, 1024, 2, latent_height, latent_width),  # 4: mid_block.resnets.1.conv2
            (batch_size, 1024, 2, latent_height, latent_width),  # 5: up_blocks.0.resnets.0.conv1
            (batch_size, 1024, 2, latent_height, latent_width),  # 6: up_blocks.0.resnets.0.conv2
            (batch_size, 1024, 2, latent_height, latent_width),  # 7: up_blocks.0.resnets.1.conv1
            (batch_size, 1024, 2, latent_height, latent_width),  # 8: up_blocks.0.resnets.1.conv2
            (batch_size, 1024, 2, latent_height, latent_width),  # 9: up_blocks.0.resnets.2.conv1
            (batch_size, 1024, 2, latent_height, latent_width),  # 10: up_blocks.0.resnets.2.conv2
            (batch_size, 1024, 2, latent_height, latent_width),  # 11: up_blocks.0.upsampler.time_conv
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 12: up_blocks.1.resnets.0.conv1
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 13: up_blocks.1.resnets.0.conv2
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 14: up_blocks.1.resnets.1.conv1
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 15: up_blocks.1.resnets.1.conv2
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 16: up_blocks.1.resnets.2.conv1
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 17: up_blocks.1.resnets.2.conv2
            (batch_size, 1024, 2, latent_height*2, latent_width*2),  # 18: up_blocks.1.upsampler.time_conv
            (batch_size, 1024, 2, latent_height*4, latent_width*4),  # 19: up_blocks.2.resnets.0.conv1
            (batch_size, 512, 2, latent_height*4, latent_width*4),  # 20: up_blocks.2.resnets.0.conv2
            (batch_size, 512, 2, latent_height*4, latent_width*4),  # 21: up_blocks.2.resnets.0.conv_shortcut
            (batch_size, 512, 2, latent_height*4, latent_width*4),  # 22: up_blocks.2.resnets.1.conv1
            (batch_size, 512, 2, latent_height*4, latent_width*4),  # 23: up_blocks.2.resnets.1.conv2
            (batch_size, 512, 2, latent_height*4, latent_width*4),  # 24: up_blocks.2.resnets.2.conv1
            (batch_size, 512, 2, latent_height*8, latent_width*8),  # 25: up_blocks.2.resnets.2.conv2
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 26: up_blocks.3.resnets.0.conv1
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 27: up_blocks.3.resnets.0.conv2
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 28: up_blocks.3.resnets.0.conv_shortcut
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 29: up_blocks.3.resnets.1.conv1
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 30: up_blocks.3.resnets.1.conv2
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 31: up_blocks.3.resnets.2.conv1
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 32: up_blocks.3.resnets.2.conv2 (dummy, not used)
            (batch_size, 12, 2, latent_height*8, latent_width*8),  # 33: conv_out (dummy, not used)
        ]

    def forward(self, x, **kwargs):
        if 'feat_cache' in kwargs:
            feat_cache = kwargs['feat_cache']

            # Check if this is a compiled TorchScript model
            is_torchscript = isinstance(self.model, torch.jit.ScriptModule)

            if is_torchscript:
                # Compiled model expects 2 frames (CACHE_T=2)
                # If we only have 1 frame, pad it by duplicating
                original_frame_count = x.shape[2]
                if original_frame_count == 1:
                    # Duplicate the frame to make it 2 frames
                    x = torch.cat([x, x], dim=2)

                if self.feat_cache_shapes is None:
                    self._init_feat_cache_shapes(x)

                # Replace None values with zero tensors
                feat_cache_fixed = []
                for i, cache in enumerate(feat_cache):
                    if cache is None and i < len(self.feat_cache_shapes):
                        feat_cache_fixed.append(torch.zeros(self.feat_cache_shapes[i], dtype=x.dtype, device=x.device))
                    else:
                        feat_cache_fixed.append(cache)

                # Pass as positional arguments for TorchScript
                output = self.model(x, feat_cache_fixed)

                # If original input was 1 frame, only return the last frame
                if original_frame_count == 1:
                    output = output[:, :, -1:, :, :]

            else:
                # Uncompiled model can handle None and keyword arguments
                output = self.model(x, feat_cache=feat_cache, **kwargs)
        else:
            output = self.model(x)
        return output

    def clear_cache(self):
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()

import torch
import math
from torch import nn

# from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape
        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])
    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)
    attn_out = torch.bmm(attention_probs, value)
    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out


# def attention_wrapper_sharded_without_swap(query, key, value):
#     bs, n_head, q_len, d_head = query.shape
#     q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
#     k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
#     v = value.clone().reshape((bs*n_head, q_len, d_head))
#     attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
#     # use_sharded_attention_kernel = True # Use "need use_sharded_attention_kernel = True" in case of trn2
#     use_sharded_attention_kernel = False # We do not "need use_sharded_attention_kernel" in case of trn1/inf2, so we could make it false
#     if use_sharded_attention_kernel:
#         # grid = (vnc(2),)
#         grid = (2,)
#         _flash_fwd_call[grid](q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
#     else:
#         _flash_fwd_call(q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
#     attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
#     return attn_output


# 问题出在attention_wrapper_sharded_without_swap函数中。错误发生在尝试reshape key tensor时，维度不匹配。
# 从错误信息和debug输出可以看到：
#     自注意力（attn1）: query, key, value 都是 [1, 5, 5376, 128]
#     交叉注意力（attn2）: query 是 [1, 5, 5376, 128]，但 key 和 value 是 [1, 5, 512, 128]
# 问题在于attention_wrapper_sharded_without_swap函数假设query和key的序列长度相同（都用q_len），但在交叉注意力中，key的序列长度是512，不是5376。
# 这里是修正后的attention_wrapper_sharded_without_swap函数：
def attention_wrapper_sharded_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]  # key的序列长度可能与query不同
    v_len = value.shape[2]  # value的序列长度
    
    # 调整reshape以适应不同的序列长度
    q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, k_len))  # 使用k_len而不是q_len
    v = value.clone().reshape((bs*n_head, v_len, d_head))  # 使用v_len
    
    attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    
    use_sharded_attention_kernel = True # Use "need use_sharded_attention_kernel = True" in case of trn2
    # use_sharded_attention_kernel = False # We do not "need use_sharded_attention_kernel" in case of trn1/inf2
    
    if use_sharded_attention_kernel:
        # grid = (vnc(2),)
        grid = (2,)
        _flash_fwd_call[grid](q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    
    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None, scale=None, enable_gqa=False):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    else:
        return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        
def attention_wrapper_for_transformer(query, key, value, attn_mask=None, dropout_p=None, is_causal=None, scale=None, enable_gqa=False):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    else:
        return attention_wrapper_sharded_without_swap(query, key, value)
        
class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)
    
    