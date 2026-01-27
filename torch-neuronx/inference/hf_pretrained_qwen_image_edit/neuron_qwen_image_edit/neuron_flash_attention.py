"""
NKI Flash Attention integration for QwenImage Transformer on Neuron.

This module provides a custom attention processor that uses Neuron's NKI Flash Attention
kernel for significantly improved performance on Trainium2.

Requirements:
- Sequence length must be divisible by 2048
- For 1024x1024 output with patch_multiplier=3: 12288 patches (12288 % 2048 = 0) ✓

Usage:
    from neuron_flash_attention import patch_transformer_with_flash_attention
    transformer = patch_transformer_with_flash_attention(transformer)
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention


def get_nki_flash_attn():
    """Get NKI Flash Attention function with proper error handling."""
    try:
        from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
        return nki_flash_attn_func
    except ImportError as e:
        raise ImportError(
            "NKI Flash Attention requires neuronx_distributed. "
            "Please install: pip install neuronx-distributed"
        ) from e


class NKIFlashAttnQwenDoubleStreamProcessor:
    """
    Custom attention processor for Qwen double-stream architecture using NKI Flash Attention.

    This replaces QwenDoubleStreamAttnProcessor2_0 with NKI Flash Attention for better
    performance on Trainium2.

    Key differences from standard processor:
    - Uses NKI Flash Attention kernel instead of torch SDPA
    - Uses transpose_nki_inputs=True for (batch, seq, heads, head_dim) layout
    - Optimized for TRN2 with lnc=2
    """

    def __init__(self, use_flash_attention: bool = True, lnc: int = 2):
        """
        Args:
            use_flash_attention: Whether to use NKI Flash Attention (True) or fallback to SDPA (False)
            lnc: Logical Neuron Core config for TRN2 (default: 2)
        """
        self.use_flash_attention = use_flash_attention
        self.lnc = lnc
        self._nki_flash_attn = None

        if use_flash_attention:
            self._nki_flash_attn = get_nki_flash_attn()

    def _flash_attention(
        self,
        query: torch.Tensor,  # (batch, seq, heads, head_dim)
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply NKI Flash Attention.

        Input shape: (batch, seq, heads, head_dim)
        Output shape: (batch, seq, heads, head_dim)

        NKI kernel internally expects (batch, heads, head_dim, seq).
        With transpose_nki_inputs=True, we provide (batch, heads, head_dim, seq).
        Output is always (batch, seq, heads, head_dim).
        """
        batch, seq, heads, head_dim = query.shape

        # Permute from (batch, seq, heads, head_dim) to (batch, heads, head_dim, seq)
        # This is what the kernel actually expects internally
        q = query.permute(0, 2, 3, 1).contiguous()
        k = key.permute(0, 2, 3, 1).contiguous()
        v = value.permute(0, 2, 3, 1).contiguous()

        # Apply NKI Flash Attention
        # Input: (batch, heads, head_dim, seq)
        # Output: (batch, seq, heads, head_dim)
        out = self._nki_flash_attn(
            q, k, v,
            lnc=self.lnc,
            causal=False,  # Image attention is not causal
            mixed_precision=True,
            dropout_p=0.0,
            transpose_nki_inputs=True,  # Input is (batch, heads, head_dim, seq)
        )

        # Output is already (batch, seq, heads, head_dim)
        return out

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback to standard SDPA."""
        # Permute to (batch, heads, seq, head_dim) for SDPA
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # Permute back to (batch, seq, heads, head_dim)
        return out.permute(0, 2, 1, 3)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass with NKI Flash Attention.

        This implements the same logic as QwenDoubleStreamAttnProcessor2_0 but uses
        NKI Flash Attention for the attention computation.
        """
        if encoder_hidden_states is None:
            raise ValueError("NKIFlashAttnQwenDoubleStreamProcessor requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE if provided
        if image_rotary_emb is not None:
            from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply attention (NKI Flash or SDPA fallback)
        if self.use_flash_attention and self._nki_flash_attn is not None:
            joint_hidden_states = self._flash_attention(joint_query, joint_key, joint_value)
        else:
            joint_hidden_states = self._sdpa_attention(joint_query, joint_key, joint_value)

        # Reshape back: (batch, seq, heads * head_dim)
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


def patch_transformer_with_flash_attention(transformer, use_flash_attention: bool = True, lnc: int = 2):
    """
    Patch a QwenImage transformer to use NKI Flash Attention.

    Args:
        transformer: QwenImageTransformer2DModel instance
        use_flash_attention: Whether to use NKI Flash Attention
        lnc: Logical Neuron Core config (2 for TRN2)

    Returns:
        The patched transformer (modified in place)

    Example:
        transformer = patch_transformer_with_flash_attention(pipe.transformer)
    """
    processor = NKIFlashAttnQwenDoubleStreamProcessor(
        use_flash_attention=use_flash_attention,
        lnc=lnc
    )

    # Count blocks patched
    num_patched = 0

    for block in transformer.transformer_blocks:
        if hasattr(block, 'attn'):
            block.attn.processor = processor
            num_patched += 1

    print(f"Patched {num_patched} transformer blocks with NKI Flash Attention (lnc={lnc})")
    return transformer


def validate_sequence_length(seq_len: int) -> bool:
    """Check if sequence length is compatible with NKI Flash Attention."""
    if seq_len % 2048 == 0:
        return True
    else:
        print(f"WARNING: Sequence length {seq_len} is not divisible by 2048.")
        print(f"  Padding will be applied, which may affect performance.")
        return False


def calculate_total_seq_len(height: int, width: int, patch_multiplier: int, text_seq_len: int = 512) -> int:
    """
    Calculate total sequence length for attention.

    Args:
        height: Output image height
        width: Output image width
        patch_multiplier: Number of temporal frames (2 for editing, 3 for 2-image merge)
        text_seq_len: Text sequence length

    Returns:
        Total sequence length (text + image patches)
    """
    latent_h = height // 8
    latent_w = width // 8
    patch_h = latent_h // 2
    patch_w = latent_w // 2
    image_patches = patch_multiplier * patch_h * patch_w
    total_seq = text_seq_len + image_patches

    print(f"Sequence length calculation:")
    print(f"  Image patches: {patch_multiplier} x {patch_h} x {patch_w} = {image_patches}")
    print(f"  Text tokens: {text_seq_len}")
    print(f"  Total: {total_seq}")
    print(f"  Divisible by 2048: {total_seq % 2048 == 0}")

    return total_seq
