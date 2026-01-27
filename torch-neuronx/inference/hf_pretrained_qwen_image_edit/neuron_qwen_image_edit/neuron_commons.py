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

    Supports two modes for Language Model:
    1. compiled_language_model: Neuron-compiled model (requires correct TP alignment)
    2. cpu_language_model: Original model on CPU (slower but avoids GQA issues)

    IMPORTANT: This wrapper COPIES necessary components and does NOT keep
    references to the original model, to avoid memory bloat.
    """
    def __init__(self, original_text_encoder, compiled_vision_encoder=None,
                 compiled_language_model=None, cpu_language_model=None,
                 image_size=448, max_seq_len=512):
        super().__init__()
        # Copy config (small object)
        self.config = original_text_encoder.config
        self.dtype = torch.bfloat16

        # IMPORTANT: Copy embed_tokens weights instead of keeping reference!
        # This allows the original model to be garbage collected.
        orig_embed = original_text_encoder.model.language_model.embed_tokens
        self.embed_tokens = nn.Embedding(
            orig_embed.num_embeddings,
            orig_embed.embedding_dim,
            padding_idx=orig_embed.padding_idx,
            dtype=torch.bfloat16
        )
        self.embed_tokens.weight.data = orig_embed.weight.data.clone().to(torch.bfloat16)
        print(f"  Copied embed_tokens: {orig_embed.num_embeddings} x {orig_embed.embedding_dim} "
              f"= {orig_embed.weight.numel() * 2 / 1e9:.2f} GB")

        # Copy visual_merger if it exists (small module)
        if hasattr(original_text_encoder.model.visual, 'merger'):
            # Deep copy the merger module
            import copy
            self.visual_merger = copy.deepcopy(original_text_encoder.model.visual.merger)
            self.visual_merger = self.visual_merger.to(torch.bfloat16)
        else:
            self.visual_merger = None

        # Compiled models
        self.compiled_vision_encoder = compiled_vision_encoder
        self.compiled_language_model = compiled_language_model

        # CPU Language Model (alternative to compiled, avoids GQA alignment issues)
        self.cpu_language_model = cpu_language_model
        self.use_cpu_language_model = cpu_language_model is not None

        # DO NOT keep original_text_encoder - it's 16+ GB!
        # self.original_text_encoder = original_text_encoder  # REMOVED!

        # Image processing parameters
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.patch_size = 14
        self.spatial_merge_size = 2

        # Calculate expected dimensions
        num_patches_per_side = image_size // self.patch_size
        self.num_image_tokens = (num_patches_per_side // self.spatial_merge_size) ** 2

        # Special token IDs from config
        self.image_token_id = getattr(self.config, 'image_token_id', 151655)
        self.vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)

    def _get_rope_index(self, input_ids, image_grid_thw, attention_mask):
        """
        Calculate 3D position_ids for M-RoPE (Multimodal RoPE).

        For multimodal input (text + images), position_ids have different patterns:
        - Text tokens: sequential positions (same for t, h, w dimensions)
        - Image tokens: 3D grid positions based on spatial layout

        This replicates the logic from Qwen2_5_VLModel.get_rope_index().
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # If no images, use simple text-only position_ids
        if image_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
            return position_ids

        # Multimodal case: need to compute proper 3D positions
        position_ids = torch.ones(3, batch_size, seq_len, dtype=torch.long, device=device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for b in range(batch_size):
            # Get non-padded tokens for this batch
            valid_mask = attention_mask[b] == 1
            valid_ids = input_ids[b][valid_mask]
            valid_len = valid_ids.shape[0]

            # Find image token positions in the valid sequence
            is_image_token = (valid_ids == self.image_token_id)
            num_actual_image_tokens = is_image_token.sum().item()

            if num_actual_image_tokens == 0:
                # No images, use sequential positions
                pos = torch.arange(valid_len, device=device)
                position_ids[:, b, valid_mask] = pos.unsqueeze(0).expand(3, -1)
                continue

            # Get grid dimensions for this image
            t, h, w = image_grid_thw[0].tolist()
            llm_grid_h = h // self.spatial_merge_size
            llm_grid_w = w // self.spatial_merge_size

            # Build position_ids token by token
            # For simplicity: text tokens get sequential positions, image tokens get grid positions
            pos_list = []
            current_pos = 0
            img_token_idx = 0

            for i in range(valid_len):
                if is_image_token[i]:
                    # Image token: use 2D grid position
                    grid_idx = img_token_idx
                    t_pos = grid_idx // (llm_grid_h * llm_grid_w)
                    remainder = grid_idx % (llm_grid_h * llm_grid_w)
                    h_pos = remainder // llm_grid_w
                    w_pos = remainder % llm_grid_w
                    # Add offset from previous text
                    pos_list.append([current_pos + t_pos, current_pos + h_pos, current_pos + w_pos])
                    img_token_idx += 1
                    # Update current_pos after last image token
                    if img_token_idx == num_actual_image_tokens:
                        current_pos = current_pos + max(t_pos, h_pos, w_pos) + 1
                else:
                    # Text token: use sequential position
                    pos_list.append([current_pos, current_pos, current_pos])
                    current_pos += 1

            # Convert to tensor
            pos_tensor = torch.tensor(pos_list, dtype=torch.long, device=device).T  # [3, valid_len]
            position_ids[:, b, valid_mask] = pos_tensor

        return position_ids

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, output_hidden_states=True, return_dict=True, **kwargs):
        """
        Forward pass combining vision encoder and language model.

        For Neuron inference, we run:
        1. Vision encoder on compiled model (or CPU fallback)
        2. Combine image embeds with text embeds
        3. Pad to max_seq_len for compiled model
        4. Language model on compiled model
        5. Remove padding from output
        """
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        # Step 1: Process images through vision encoder
        if pixel_values is not None:
            # Ensure pixel_values is bfloat16 and correct shape
            pixel_values = pixel_values.to(torch.bfloat16)

            # Use compiled vision encoder or CPU fallback
            if self.compiled_vision_encoder is not None:
                # Check if we need to pad/reshape to expected size
                expected_patches = (self.image_size // self.patch_size) ** 2  # 1024 for 448x448
                actual_patches = pixel_values.shape[0]

                if actual_patches != expected_patches:
                    # Pad or truncate to expected size
                    if actual_patches < expected_patches:
                        # Pad with zeros
                        padding = torch.zeros(
                            expected_patches - actual_patches,
                            pixel_values.shape[1],
                            dtype=pixel_values.dtype,
                            device=pixel_values.device
                        )
                        pixel_values = torch.cat([pixel_values, padding], dim=0)
                    else:
                        # Truncate
                        pixel_values = pixel_values[:expected_patches]

                    # Update image_grid_thw to match
                    grid_size = self.image_size // self.patch_size
                    image_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                image_embeds = self.compiled_vision_encoder(pixel_values, image_grid_thw)
                # Note: merger is already included in compiled_vision_encoder
            else:
                # No vision encoder compiled - this should not happen in full Neuron mode
                raise RuntimeError(
                    "Vision encoder not compiled! Please compile the vision encoder first:\n"
                    "  python neuron_qwen_image_edit/compile_text_encoder.py --vision_only"
                )
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

        # Step 4: Calculate 3D position_ids for M-RoPE (required by Qwen2.5-VL)
        # For multimodal input (text + images), position_ids have special patterns:
        # - Text tokens: sequential positions (same for t, h, w dimensions)
        # - Image tokens: 3D grid positions based on spatial layout
        position_ids = self._get_rope_index(input_ids, image_grid_thw, attention_mask)

        # Step 5: Run language model (CPU or compiled)
        if self.use_cpu_language_model:
            # CPU Language Model mode - no padding needed, handles dynamic sequence lengths
            # This avoids GQA alignment issues that occur with TP != 4
            with torch.no_grad():
                cpu_outputs = self.cpu_language_model(
                    inputs_embeds=inputs_embeds.to(torch.bfloat16),
                    attention_mask=attention_mask,
                    position_ids=position_ids,  # Pass 3D position_ids for M-RoPE
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = cpu_outputs.last_hidden_state

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states

        elif self.compiled_language_model is not None:
            # Neuron compiled Language Model mode - requires fixed sequence length
            original_seq_len = inputs_embeds.shape[1]
            hidden_size = inputs_embeds.shape[2]

            if original_seq_len < self.max_seq_len:
                # Pad inputs_embeds with zeros
                pad_len = self.max_seq_len - original_seq_len
                embed_padding = torch.zeros(
                    batch_size, pad_len, hidden_size,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                inputs_embeds = torch.cat([inputs_embeds, embed_padding], dim=1)

                # Pad attention_mask with zeros (masked positions)
                if attention_mask is not None:
                    mask_padding = torch.zeros(
                        batch_size, pad_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
            elif original_seq_len > self.max_seq_len:
                # Truncate if too long
                print(f"  WARNING: Sequence length {original_seq_len} > max_seq_len {self.max_seq_len}, truncating")
                inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_seq_len]
                original_seq_len = self.max_seq_len

            # Run compiled language model
            hidden_states = self.compiled_language_model(inputs_embeds, attention_mask)

            # Remove padding from output (restore original sequence length)
            hidden_states = hidden_states[:, :original_seq_len, :]

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states

        else:
            # No language model available
            raise RuntimeError(
                "No language model available! Please either:\n"
                "1. Compile language model: python neuron_qwen_image_edit/compile_text_encoder.py --language_only\n"
                "2. Use CPU language model by passing cpu_language_model to NeuronTextEncoderWrapper"
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

    Supports:
    - Grouped Query Attention (GQA) where num_kv_heads < num_q_heads
    - Causal masking when is_causal=True
    - Explicit attention masks (attn_mask)
    """
    orig_shape = None
    orig_query_shape = query.shape
    q_len = query.shape[-2]
    kv_len = key.shape[-2]

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

    # Compute attention scores: [batch*heads, q_len, kv_len]
    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * scale

    # Apply causal mask if requested
    if is_causal:
        # Create causal mask: positions above the main diagonal are masked (-inf)
        # Shape: (q_len, kv_len)
        # Use torch.where to avoid NaN from 0 * -inf
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=attention_scores.device),
            diagonal=1
        )
        causal_mask = torch.where(
            causal_mask == 1,
            torch.tensor(float('-inf'), dtype=attention_scores.dtype, device=attention_scores.device),
            torch.tensor(0.0, dtype=attention_scores.dtype, device=attention_scores.device)
        )
        attention_scores = attention_scores + causal_mask

    # Apply explicit attention mask if provided
    if attn_mask is not None:
        # attn_mask can be:
        # - 2D: (q_len, kv_len) - applied to all batches/heads
        # - 3D: (batch*heads, q_len, kv_len) - per-head mask
        # - 4D: (batch, heads, q_len, kv_len) - full mask
        if attn_mask.dim() == 4:
            # Reshape 4D mask to 3D
            attn_mask = attn_mask.reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        elif attn_mask.dim() == 2:
            # Broadcast 2D mask
            attn_mask = attn_mask.unsqueeze(0)

        # Convert boolean mask to additive mask if needed
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        attention_scores = attention_scores + attn_mask.to(attention_scores.dtype)

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
    """Attention wrapper for text encoder.

    Always uses our custom implementation for better Neuron tracing compatibility.
    The custom implementation supports:
    - Causal masking (is_causal=True)
    - Explicit attention masks (attn_mask)
    - GQA (handled by repeat_kv in model's forward, but we handle leftovers)
    """
    # Always use our custom implementation for Neuron compatibility
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
