import torch
import math
from torch import nn
from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Try to import NKI kernel, but don't fail if not available
try:
    import neuronxcc.nki as nki
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    _flash_fwd_call = nki.jit()(attention_isa_kernel)
    NKI_AVAILABLE = True
except ImportError:
    _flash_fwd_call = None
    NKI_AVAILABLE = False


class InferenceTextEncoderWrapper(nn.Module):
    """Wrapper for Qwen2.5-VL text encoder for inference on Neuron."""
    def __init__(self, dtype, text_encoder: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.dtype = dtype
        self.device = text_encoder.device
        self.text_encoder = text_encoder
        self.config = text_encoder.config

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                image_grid_thw=None, **kwargs):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **kwargs
        )
        return outputs


class NeuronTextEncoderWrapper(nn.Module):
    """
    Wrapper for compiled Qwen2.5-VL text encoder on Neuron.

    Combines separately compiled vision encoder and language model.
    This wrapper handles the embedding combination logic that normally
    happens inside the original text encoder.
    """
    def __init__(self, original_text_encoder, compiled_vision_encoder=None,
                 compiled_language_model=None, image_size=448, max_seq_len=512):
        super().__init__()
        self.config = original_text_encoder.config
        self.dtype = torch.bfloat16

        # Store original components for embedding operations
        self.embed_tokens = original_text_encoder.model.language_model.embed_tokens
        self.visual_merger = original_text_encoder.model.visual.merger if hasattr(original_text_encoder.model.visual, 'merger') else None

        # Compiled models
        self.compiled_vision_encoder = compiled_vision_encoder
        self.compiled_language_model = compiled_language_model

        # Keep original for fallback
        self.original_text_encoder = original_text_encoder

        # Image processing parameters
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.patch_size = 14
        self.spatial_merge_size = 2

        # Calculate expected dimensions
        num_patches_per_side = image_size // self.patch_size
        self.num_image_tokens = (num_patches_per_side // self.spatial_merge_size) ** 2

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, output_hidden_states=True, return_dict=True, **kwargs):
        """
        Forward pass combining vision encoder and language model.

        For Neuron inference, we run:
        1. Vision encoder on compiled model
        2. Combine image embeds with text embeds
        3. Language model on compiled model
        """
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        # If we have both compiled models, use them
        if self.compiled_vision_encoder is not None and self.compiled_language_model is not None:
            # Step 1: Process images through vision encoder
            if pixel_values is not None:
                image_embeds = self.compiled_vision_encoder(pixel_values, image_grid_thw)

                # Apply merger if exists (spatial merge)
                if self.visual_merger is not None:
                    image_embeds = self.visual_merger(image_embeds)
            else:
                image_embeds = None

            # Step 2: Get text embeddings
            text_embeds = self.embed_tokens(input_ids)

            # Step 3: Combine embeddings
            # Find image token positions and replace with image embeddings
            if image_embeds is not None:
                # The image token ID in Qwen2.5-VL
                image_token_id = self.config.image_token_id if hasattr(self.config, 'image_token_id') else 151655

                # Create combined embeddings
                inputs_embeds = self._merge_embeddings(
                    text_embeds, image_embeds, input_ids, image_token_id
                )
            else:
                inputs_embeds = text_embeds

            # Step 4: Run language model
            hidden_states = self.compiled_language_model(inputs_embeds, attention_mask)

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states
        else:
            # Fallback to original
            return self.original_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

    def _merge_embeddings(self, text_embeds, image_embeds, input_ids, image_token_id):
        """Merge text and image embeddings at image token positions."""
        batch_size, seq_len, hidden_size = text_embeds.shape

        # Find positions of image tokens
        image_mask = (input_ids == image_token_id)

        # Replace image token embeddings with actual image embeddings
        inputs_embeds = text_embeds.clone()

        for b in range(batch_size):
            image_positions = torch.where(image_mask[b])[0]
            if len(image_positions) > 0 and image_embeds is not None:
                num_image_tokens = min(len(image_positions), image_embeds.shape[0])
                inputs_embeds[b, image_positions[:num_image_tokens]] = image_embeds[:num_image_tokens]

        return inputs_embeds


class InferenceTransformerWrapper(nn.Module):
    """Wrapper for QwenImageTransformer2DModel for inference on Neuron."""
    def __init__(self, transformer: QwenImageTransformer2DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, encoder_attention_mask=None,
                pooled_projections=None, return_dict=False, **kwargs):
        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            return_dict=return_dict,
        )
        return output


class SimpleWrapper(nn.Module):
    """Simple wrapper for VAE decoder and other modules."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class f32Wrapper(nn.Module):
    """Wrapper to run normalization layers in float32 for numerical stability."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=None, is_causal=None, scale=None):
    """Custom scaled dot product attention optimized for Neuron.

    Supports Grouped Query Attention (GQA) where num_kv_heads < num_q_heads.
    """
    orig_shape = None
    orig_query_shape = query.shape

    if len(query.shape) == 4:
        orig_shape = query.shape
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, _, _ = key.shape

        # Handle GQA: repeat K/V heads to match Q heads
        if num_kv_heads != num_q_heads:
            num_groups = num_q_heads // num_kv_heads
            # Repeat K and V along head dimension
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])

    # Use provided scale or default to 1/sqrt(d_k)
    if scale is None:
        scale = 1 / math.sqrt(query.size(-1))

    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * scale
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        attention_probs = attention_scores.softmax(dim=-1)

    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out


def attention_wrapper_sharded_without_swap(query, key, value):
    """Sharded attention wrapper using NKI kernel for trn2.

    Note: This kernel requires Q, K, V to have the same sequence length.
    For cross-attention with different lengths, fall back to basic attention.
    """
    bs, n_head, q_len, d_head = query.shape
    _, _, kv_len, _ = key.shape

    # NKI kernel requires same sequence length for Q, K, V and NKI must be available
    if q_len != kv_len or not NKI_AVAILABLE or _flash_fwd_call is None:
        # Fall back to basic attention
        return neuron_scaled_dot_product_attention(query, key, value)

    q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, kv_len))
    v = value.clone().reshape((bs*n_head, kv_len, d_head))
    attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    # Use sharded attention kernel for trn2
    use_sharded_attention_kernel = True
    if use_sharded_attention_kernel:
        grid = (2,)
        _flash_fwd_call[grid](q, k, v, 0.117, attn_output,
                              kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, 0.117, attn_output,
                        kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


# Store original SDPA function
sdpa_original = torch.nn.functional.scaled_dot_product_attention


def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None,
                      scale=None, enable_gqa=False):
    """Attention wrapper for text encoder (uses mask when provided)."""
    if attn_mask is not None:
        # Pass through to original SDPA which handles GQA
        return sdpa_original(query, key, value, attn_mask=attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                            enable_gqa=enable_gqa)
    else:
        return neuron_scaled_dot_product_attention(query, key, value,
                                                   attn_mask=attn_mask,
                                                   dropout_p=dropout_p,
                                                   is_causal=is_causal,
                                                   scale=scale)


def attention_wrapper_for_transformer(query, key, value, attn_mask=None,
                                       dropout_p=None, is_causal=None,
                                       scale=None):
    """Attention wrapper for transformer.

    Uses basic softmax attention for better compatibility during compilation.
    NKI kernel can be enabled later for performance optimization.
    """
    # For now, use basic attention for better compatibility
    # NKI kernel has shape constraints that may not work with all attention patterns
    return neuron_scaled_dot_product_attention(query, key, value,
                                               attn_mask=attn_mask,
                                               dropout_p=dropout_p,
                                               is_causal=is_causal)
