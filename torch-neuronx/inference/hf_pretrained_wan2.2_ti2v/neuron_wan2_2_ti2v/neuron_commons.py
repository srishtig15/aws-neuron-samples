import time
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


class EncoderWrapperNoCache(nn.Module):
    """Wrapper for compiled encoder that was compiled WITHOUT feat_cache

    This wrapper ignores feat_cache and feat_idx arguments since the encoder
    was compiled without temporal caching support.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, feat_cache=None, feat_idx=None, **kwargs):
        # Ignore feat_cache and feat_idx - compiled encoder doesn't use them
        output = self.model(x)
        return output

    def clear_cache(self):
        # No cache to clear
        pass


class EncoderWrapper(nn.Module):
    """Specialized wrapper for VAE encoder that handles TorchScript feat_cache compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Store the expected feat_cache shapes for compiled encoder
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x (AFTER patchify)"""
        batch_size = x.shape[0]
        # x is AFTER patchify: (batch, 12, frames, patchified_height, patchified_width)
        # For 512x512 input with patch_size=2: (batch, 12, frames, 256, 256)
        patchified_height = x.shape[3]
        patchified_width = x.shape[4]

        # Create feat_cache with correct shapes (EXACTLY matching compile_encoder.py)
        # IMPORTANT: feat_cache stores INPUT shape to each conv layer
        # All feat_cache tensors have time dimension of 2 (CACHE_T=2)
        # Encoder downsamples spatially from patchified resolution: 256 -> 128 -> 64 -> 32
        self.feat_cache_shapes = [
            # conv_in: 12 → 160
            (batch_size, 12, 2, patchified_height, patchified_width),
            # down_blocks.0: 160 channels throughout, 256x256
            (batch_size, 160, 2, patchified_height, patchified_width),  # resnets.0.conv1 (160→160)
            (batch_size, 160, 2, patchified_height, patchified_width),  # resnets.0.conv2 (160→160)
            (batch_size, 160, 2, patchified_height, patchified_width),  # resnets.1.conv1 (160→160)
            (batch_size, 160, 2, patchified_height, patchified_width),  # resnets.1.conv2 (160→160)
            # down_blocks.1: 160 → 320 channel increase, 128x128
            # NOTE: conv_shortcut is NOT in feat_cache (called without feat_cache argument)
            (batch_size, 160, 2, patchified_height//2, patchified_width//2),  # resnets.0.conv1 (160→320)
            (batch_size, 320, 2, patchified_height//2, patchified_width//2),  # resnets.0.conv2 (320→320)
            (batch_size, 320, 2, patchified_height//2, patchified_width//2),  # resnets.1.conv1 (320→320)
            (batch_size, 320, 2, patchified_height//2, patchified_width//2),  # resnets.1.conv2 (320→320)
            (batch_size, 320, 2, patchified_height//4, patchified_width//4),  # downsampler.time_conv (320→320) - AFTER spatial downsample!
            # down_blocks.2: 320 → 640 channel increase, 64x64
            # NOTE: conv_shortcut is NOT in feat_cache (called without feat_cache argument)
            (batch_size, 320, 2, patchified_height//4, patchified_width//4),  # resnets.0.conv1 (320→640)
            (batch_size, 640, 2, patchified_height//4, patchified_width//4),  # resnets.0.conv2 (640→640)
            (batch_size, 640, 2, patchified_height//4, patchified_width//4),  # resnets.1.conv1 (640→640)
            (batch_size, 640, 2, patchified_height//4, patchified_width//4),  # resnets.1.conv2 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # downsampler.time_conv (640→640) - AFTER spatial downsample!
            # down_blocks.3: 640 channels throughout, 32x32
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.0.conv1 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.0.conv2 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.1.conv1 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.1.conv2 (640→640)
            # mid_block: 640 channels throughout, 32x32
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.0.conv1 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.0.conv2 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.1.conv1 (640→640)
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # resnets.1.conv2 (640→640)
            # conv_out: 640 → 96
            (batch_size, 640, 2, patchified_height//8, patchified_width//8),  # conv_out (640→96)
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

                # Propagate updates from feat_cache_fixed back to original feat_cache
                # This is crucial for temporal caching to work across iterations
                for i in range(len(feat_cache)):
                    feat_cache[i] = feat_cache_fixed[i]

                # Encoder processes 2 input frames -> outputs latents with temporal downsampling
                # For 2 input frames -> 1 latent frame (4x temporal downsampling)
                # If original input was 1 frame (duplicated to 2), we don't need to adjust output
                # because the encoder naturally outputs the correct number of latent frames

            else:
                # Uncompiled model can handle None and keyword arguments
                output = self.model(x, feat_cache=feat_cache, **kwargs)
        else:
            output = self.model(x)
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

                # Propagate updates from feat_cache_fixed back to original feat_cache
                # This is crucial for temporal caching to work across iterations
                for i in range(len(feat_cache)):
                    feat_cache[i] = feat_cache_fixed[i]

                # If original input was 1 frame, decoder outputs 8 frames (2 latent × 4x upsampling)
                # We take the last 4 frames (corresponding to the duplicated latent frame)
                if original_frame_count == 1:
                    # Decoder does 4x temporal upsampling: 1 latent frame → 4 output frames
                    # Since we duplicated to 2 frames: 2 latent frames → 8 output frames
                    # Take the last 4 frames (from the second, duplicated latent frame)
                    output = output[:, :, -4:, :, :]

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


class DecoderWrapperV2(nn.Module):
    """
    Wrapper for V2 compiled VAE decoder using NxDModel.

    The V2 compiled decoder accepts 34 individual feat_cache tensors as arguments
    instead of a list, because ModelBuilder V2 API requires all inputs to be tensors.
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, original_decoder):
        super().__init__()
        self.original_decoder = original_decoder  # Keep reference for config
        self.nxd_model = None  # Will be set after loading
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x"""
        batch_size = x.shape[0]
        latent_height = x.shape[3]
        latent_width = x.shape[4]

        # Create feat_cache shapes (matching compile_decoder_v2.py)
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
            (batch_size, 256, 2, latent_height*8, latent_width*8),  # 32: up_blocks.3.resnets.2.conv2 (dummy)
            (batch_size, 12, 2, latent_height*8, latent_width*8),  # 33: conv_out (dummy)
        ]

    def forward(self, x, **kwargs):
        if 'feat_cache' not in kwargs:
            # No feat_cache, use original decoder
            return self.original_decoder(x)

        feat_cache = kwargs['feat_cache']

        # Compiled model expects 2 frames (CACHE_T=2)
        original_frame_count = x.shape[2]
        if original_frame_count == 1:
            x = torch.cat([x, x], dim=2)

        if self.feat_cache_shapes is None:
            self._init_feat_cache_shapes(x)

        # Prepare feat_cache tensors - replace None with zeros
        feat_cache_tensors = []
        for i in range(self.NUM_FEAT_CACHE):
            if i < len(feat_cache) and feat_cache[i] is not None:
                feat_cache_tensors.append(feat_cache[i])
            else:
                feat_cache_tensors.append(
                    torch.zeros(self.feat_cache_shapes[i], dtype=x.dtype, device=x.device)
                )

        # Call NxDModel with individual feat_cache arguments
        output = self.nxd_model(
            x,
            feat_cache_tensors[0], feat_cache_tensors[1], feat_cache_tensors[2],
            feat_cache_tensors[3], feat_cache_tensors[4], feat_cache_tensors[5],
            feat_cache_tensors[6], feat_cache_tensors[7], feat_cache_tensors[8],
            feat_cache_tensors[9], feat_cache_tensors[10], feat_cache_tensors[11],
            feat_cache_tensors[12], feat_cache_tensors[13], feat_cache_tensors[14],
            feat_cache_tensors[15], feat_cache_tensors[16], feat_cache_tensors[17],
            feat_cache_tensors[18], feat_cache_tensors[19], feat_cache_tensors[20],
            feat_cache_tensors[21], feat_cache_tensors[22], feat_cache_tensors[23],
            feat_cache_tensors[24], feat_cache_tensors[25], feat_cache_tensors[26],
            feat_cache_tensors[27], feat_cache_tensors[28], feat_cache_tensors[29],
            feat_cache_tensors[30], feat_cache_tensors[31], feat_cache_tensors[32],
            feat_cache_tensors[33],
        )

        # Handle tuple return
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Propagate updates back to original feat_cache
        for i in range(min(len(feat_cache), self.NUM_FEAT_CACHE)):
            feat_cache[i] = feat_cache_tensors[i]

        # If original input was 1 frame, take last 4 frames
        if original_frame_count == 1:
            output = output[:, :, -4:, :, :]

        return output

    def clear_cache(self):
        pass


class PostQuantConvWrapperV2(nn.Module):
    """Wrapper for V2 compiled post_quant_conv using NxDModel."""

    def __init__(self, original_conv):
        super().__init__()
        self.original_conv = original_conv
        self.nxd_model = None  # Will be set after loading

    def forward(self, x, **kwargs):
        output = self.nxd_model(x)
        # Handle tuple return
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def clear_cache(self):
        pass


class EncoderWrapperV3(nn.Module):
    """
    Wrapper for V3 compiled VAE encoder (bfloat16, torch_neuronx.trace).

    The compiled model takes post-patchify input directly: (B, 12, T, 256, 256).
    This matches what _encode() passes to the encoder after patchify().

    Handles:
    - bfloat16 conversion (matching compiled dtype)
    - Ignoring feat_cache/feat_idx arguments from the _encode() loop
    """

    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.model = None  # Will be set via torch.jit.load()

    def forward(self, x, feat_cache=None, feat_idx=None, **kwargs):
        # x is patchified: (B, 12, T, 256, 256) — passed directly to compiled model
        output = self.model(x.to(torch.bfloat16))

        # Handle tuple return
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Convert back to float32 for pipeline
        return output.to(torch.float32)

    def clear_cache(self):
        pass


class QuantConvWrapperV3(nn.Module):
    """Wrapper for V3 compiled quant_conv (bfloat16, torch_neuronx.trace)."""

    def __init__(self, original_conv):
        super().__init__()
        self.original_conv = original_conv
        self.model = None  # Will be set via torch.jit.load()

    def forward(self, x, **kwargs):
        output = self.model(x.to(torch.bfloat16))
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.to(torch.float32)

    def clear_cache(self):
        pass


class DecoderWrapperV3(nn.Module):
    """
    Wrapper for V3 compiled VAE decoder (bfloat16) using NxDModel.

    The V3 decoder is compiled in bfloat16 for 2x memory bandwidth reduction.
    This wrapper handles dtype conversion: float32 input -> bfloat16 -> decoder -> float32 output.
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, original_decoder):
        super().__init__()
        self.original_decoder = original_decoder
        self.nxd_model = None
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x (after padding to 2 frames)."""
        batch_size = x.shape[0]
        latent_height = x.shape[3]
        latent_width = x.shape[4]

        self.feat_cache_shapes = [
            (batch_size, 48, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height, latent_width),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*2, latent_width*2),
            (batch_size, 1024, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*4, latent_width*4),
            (batch_size, 512, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 256, 2, latent_height*8, latent_width*8),
            (batch_size, 12, 2, latent_height*8, latent_width*8),
        ]

    def forward(self, x, **kwargs):
        if 'feat_cache' not in kwargs:
            return self.original_decoder(x)

        feat_cache = kwargs['feat_cache']

        # Compiled model expects 2 frames (CACHE_T=2)
        original_frame_count = x.shape[2]
        if original_frame_count == 1:
            x = torch.cat([x, x], dim=2)

        if self.feat_cache_shapes is None:
            self._init_feat_cache_shapes(x)

        # Convert input to bfloat16 (decoder compiled in bfloat16)
        x_bf16 = x.to(torch.bfloat16)

        # Prepare feat_cache tensors in bfloat16
        feat_cache_tensors = []
        for i in range(self.NUM_FEAT_CACHE):
            if i < len(feat_cache) and feat_cache[i] is not None:
                feat_cache_tensors.append(feat_cache[i].to(torch.bfloat16))
            else:
                feat_cache_tensors.append(
                    torch.zeros(self.feat_cache_shapes[i], dtype=torch.bfloat16)
                )

        # Call NxDModel
        output = self.nxd_model(
            x_bf16,
            feat_cache_tensors[0], feat_cache_tensors[1], feat_cache_tensors[2],
            feat_cache_tensors[3], feat_cache_tensors[4], feat_cache_tensors[5],
            feat_cache_tensors[6], feat_cache_tensors[7], feat_cache_tensors[8],
            feat_cache_tensors[9], feat_cache_tensors[10], feat_cache_tensors[11],
            feat_cache_tensors[12], feat_cache_tensors[13], feat_cache_tensors[14],
            feat_cache_tensors[15], feat_cache_tensors[16], feat_cache_tensors[17],
            feat_cache_tensors[18], feat_cache_tensors[19], feat_cache_tensors[20],
            feat_cache_tensors[21], feat_cache_tensors[22], feat_cache_tensors[23],
            feat_cache_tensors[24], feat_cache_tensors[25], feat_cache_tensors[26],
            feat_cache_tensors[27], feat_cache_tensors[28], feat_cache_tensors[29],
            feat_cache_tensors[30], feat_cache_tensors[31], feat_cache_tensors[32],
            feat_cache_tensors[33],
        )

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Convert output back to float32
        output = output.to(torch.float32)

        # Propagate bfloat16 cache back (keep as bfloat16 for next iteration)
        for i in range(min(len(feat_cache), self.NUM_FEAT_CACHE)):
            feat_cache[i] = feat_cache_tensors[i]

        # If original input was 1 frame, take last 4 frames
        if original_frame_count == 1:
            output = output[:, :, -4:, :, :]

        return output

    def clear_cache(self):
        pass


class DecoderWrapperV3NoCache(nn.Module):
    """
    Wrapper for V3 NoCache compiled decoder.

    The compiled model takes only x as input (no feat_cache arguments).
    feat_cache is internalized as registered buffers (zeros, loaded once to device).

    This eliminates ~960MB per-call data transfer. Only x (~300KB) is transferred.
    """

    def __init__(self, original_decoder, decoder_frames=2):
        super().__init__()
        self.original_decoder = original_decoder
        self.decoder_frames = decoder_frames
        self.nxd_model = None

    def forward(self, x, **kwargs):
        if 'feat_cache' not in kwargs:
            return self.original_decoder(x)

        _t0 = time.time()

        # Determine original frame count before padding
        original_frame_count = x.shape[2]

        # Pad temporal dimension to decoder_frames if needed
        if x.shape[2] < self.decoder_frames:
            pad_frames = self.decoder_frames - x.shape[2]
            x = torch.cat([x] + [x[:, :, -1:]] * pad_frames, dim=2)

        # Convert to bfloat16 for the compiled decoder
        x_bf16 = x.to(torch.bfloat16)

        _t1 = time.time()

        # NoCache: only pass x as input (1 argument, ~300KB)
        # vs V3 cache: x + 34 feat_cache tensors (35 arguments, ~960MB)
        output = self.nxd_model(x_bf16)

        _t2 = time.time()

        # Convert back to float32 and trim to original frame count
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output.to(torch.float32)

        # Trim padded frames: output temporal = original_frame_count * 4 (due to upsampling)
        output_frames = original_frame_count * 4
        if output.shape[2] > output_frames:
            output = output[:, :, :output_frames]

        _t3 = time.time()

        print(f"[nocache] prep={_t1-_t0:.4f}s nxd_model={_t2-_t1:.4f}s postproc={_t3-_t2:.4f}s total={_t3-_t0:.4f}s frames={original_frame_count}")

        return output

    def clear_cache(self):
        pass


