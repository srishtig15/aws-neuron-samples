"""
Wrapper classes for Wan2.2 inference using Model Builder V2 API.

These wrappers work with NxDModel instead of TorchScript models.
"""
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from transformers.models.umt5 import UMT5EncoderModel
import torch
from torch import nn
from types import SimpleNamespace


class InferenceTextEncoderWrapperV2(nn.Module):
    """Wrapper for text encoder with NxDModel V2 API."""

    def __init__(self, dtype, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        self.t = t

    def forward(self, text_input_ids, attention_mask=None):
        # Try tagged method first, fall back to forward()
        if hasattr(self.t, 'encode'):
            result = self.t.encode(
                text_input_ids=text_input_ids,
                attention_mask=attention_mask
            )
        else:
            result = self.t(text_input_ids, attention_mask)

        # Handle different return types
        if isinstance(result, dict):
            last_hidden_state = result.get('last_hidden_state', result.get(0))
        elif isinstance(result, (tuple, list)):
            last_hidden_state = result[0]
        else:
            last_hidden_state = result

        return SimpleNamespace(last_hidden_state=last_hidden_state.to(self.dtype))


class InferenceTransformerWrapperV2(nn.Module):
    """Wrapper for transformer with NxDModel V2 API."""

    def __init__(self, transformer: WanTransformer3DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
        self.cache_context = transformer.cache_context

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, return_dict=False, **kwargs):
        # Try tagged method first, fall back to forward()
        if hasattr(self.transformer, 'inference'):
            output = self.transformer.inference(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states
            )
        else:
            output = self.transformer(
                hidden_states,
                timestep,
                encoder_hidden_states
            )

        # Handle tuple return (NxDModel may return tuple)
        if isinstance(output, (tuple, list)):
            output = output[0]

        return output


class SimpleWrapperV2(nn.Module):
    """Simple wrapper for post_quant_conv with NxDModel V2 API."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, **kwargs):
        # Try tagged method first, fall back to forward()
        if hasattr(self.model, 'conv'):
            output = self.model.conv(x=x)
        else:
            output = self.model(x)

        # Handle tuple return
        if isinstance(output, (tuple, list)):
            output = output[0]

        return output

    def clear_cache(self):
        pass


class DecoderWrapperV2(nn.Module):
    """Wrapper for VAE decoder with NxDModel V2 API.

    Handles feat_cache compatibility for temporal caching.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, x):
        """Initialize feat_cache shapes based on input x."""
        batch_size = x.shape[0]
        latent_height = x.shape[3]
        latent_width = x.shape[4]

        # Match compile_decoder_v2.py feat_cache shapes
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
        if 'feat_cache' in kwargs:
            feat_cache = kwargs['feat_cache']

            # Check if using NxDModel
            is_nxd_model = hasattr(self.model, 'decode')

            if is_nxd_model:
                # NxDModel compiled with CACHE_T=2 frames
                original_frame_count = x.shape[2]
                if original_frame_count == 1:
                    x = torch.cat([x, x], dim=2)

                if self.feat_cache_shapes is None:
                    self._init_feat_cache_shapes(x)

                # Replace None values with zero tensors
                feat_cache_fixed = []
                for i, cache in enumerate(feat_cache):
                    if cache is None and i < len(self.feat_cache_shapes):
                        feat_cache_fixed.append(
                            torch.zeros(self.feat_cache_shapes[i], dtype=x.dtype, device=x.device)
                        )
                    else:
                        feat_cache_fixed.append(cache)

                # Call with tag 'decode'
                output = self.model.decode(x=x, feat_cache=feat_cache_fixed)

                # Handle tuple return
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # Propagate feat_cache updates
                for i in range(len(feat_cache)):
                    feat_cache[i] = feat_cache_fixed[i]

                # Handle frame count adjustment
                if original_frame_count == 1:
                    output = output[:, :, -4:, :, :]
            else:
                # Non-compiled model
                output = self.model(x, feat_cache=feat_cache, **kwargs)
        else:
            if hasattr(self.model, 'decode'):
                output = self.model.decode(x=x)
                if isinstance(output, (tuple, list)):
                    output = output[0]
            else:
                output = self.model(x)

        return output

    def clear_cache(self):
        pass
