# Rolling Cache VAE Decoder - Design & Principles

## Overview

The Wan2.2 VAE decoder uses **3D causal convolutions** (`WanCausalConv3d`) that require temporal context from previous frames. The diffusers reference implementation processes frames one-by-one, carrying a `feat_cache` list across calls. On Neuron (Trainium2), we compile the decoder as an `NxDModel` and process multiple frames per call. This document explains how the **rolling cache** mechanism preserves temporal coherence.

## 1. How diffusers Decodes Video (Reference)

The native `AutoencoderKLWan._decode()` processes latent frames **one at a time**:

```python
# diffusers AutoencoderKLWan._decode()
self.clear_cache()                    # feat_cache = [None] * 34
for i in range(num_frame):            # 21 latent frames, one by one
    self._conv_idx = [0]
    if i == 0:
        out = self.decoder(x[:,:,0:1,:,:], feat_cache=self._feat_map, first_chunk=True)
    else:
        out_ = self.decoder(x[:,:,i:i+1,:,:], feat_cache=self._feat_map)
        out = cat([out, out_], dim=2)
```

The key: `feat_cache` is a **shared mutable list of 34 tensors**, updated **in-place** by every conv layer during each call, and **carried across** all 21 calls.

## 2. feat_cache: The Temporal Context Mechanism

### CACHE_T = 2

Every `WanCausalConv3d` in the decoder follows this pattern:

```python
# Inside WanResidualBlock.forward():
idx = feat_idx[0]

# 1. Save current input's last CACHE_T=2 frames as new cache
cache_x = x[:, :, -2:, :, :].clone()

# 2. If cache has fewer than 2 frames, prepend from previous cache
if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
    cache_x = cat([feat_cache[idx][:,:,-1,:,:].unsqueeze(2), cache_x], dim=2)

# 3. Pass OLD cache to CausalConv3d (prepended as temporal prefix)
x = self.conv1(x, feat_cache[idx])

# 4. Store NEW cache for next call
feat_cache[idx] = cache_x
feat_idx[0] += 1
```

Inside `WanCausalConv3d.forward()`:
```python
def forward(self, x, cache_x=None):
    if cache_x is not None and self._padding[4] > 0:
        x = torch.cat([cache_x, x], dim=2)  # prepend temporal context
        padding[4] -= cache_x.shape[2]       # reduce temporal padding
    x = F.pad(x, padding)
    return super().forward(x)                 # standard Conv3D
```

**In short**: each conv layer reads the previous call's cache (temporal context), does the convolution with that context prepended, then saves the current frame as cache for the next call.

### The 34 Cache Slots

The decoder has 34 `WanCausalConv3d` layers, each with its own cache slot. `feat_idx` is a `[0]` list (mutable reference) that increments after each conv, ensuring sequential access:

**T2V-A14B (z_dim=16, 480x832)**:
```
Index  Location                    Channels  Spatial
─────  ──────────────────────────  ────────  ───────
 0     conv_in                     16        lh × lw
 1-2   mid_block resnet_0 c1,c2   384       lh × lw
 3-4   mid_block resnet_1 c1,c2   384       lh × lw
 5-10  up_block_0 resnet×3 c1,c2  384       lh × lw
 11    up_block_0 upsampler (t×2)  384       lh × lw
 12-17 up_block_1 resnet×3 c1,c2  384       2lh × 2lw
 18    up_block_1 upsampler (t×2)  384       2lh × 2lw
 19-24 up_block_2 resnet×3 c1,c2  192       4lh × 4lw
 25-30 up_block_3 resnet×3 c1,c2  96        8lh × 8lw
 31    conv_out                    96        8lh × 8lw
 32-33 placeholders                96/12     8lh × 8lw
```

**TI2V-5B (z_dim=48, 512x512)**: Same structure but with channels 48/1024/512/256/12.

## 3. The Problem: NxDModel Doesn't Preserve In-Place Updates

When we compile the decoder with `NxDModel` (ModelBuilder), the compiled graph runs on Neuron device. The in-place `feat_cache[idx] = cache_x` updates happen **inside the compiled graph**, but NxDModel does **not** persist these changes between calls.

This leads to two compilation strategies:

## 4. NoCache Mode (feat_cache as Registered Buffers)

```python
class DecoderWrapperNoCache(nn.Module):
    def __init__(self, decoder, feat_cache_shapes):
        for i, shape in enumerate(feat_cache_shapes):
            self.register_buffer(f'feat_cache_{i}', torch.zeros(shape))

    def forward(self, x):                    # Only 1 input
        feat_cache = [self.feat_cache_0, self.feat_cache_1, ...]
        return self.decoder(x, feat_cache)   # Only 1 output
```

**Behavior**:
```
Chunk 0: decoder(frames[0:2], zeros)  → video[0:8],  cache updated internally but LOST
Chunk 1: decoder(frames[2:4], zeros)  → video[8:16], cache updated internally but LOST
Chunk 2: decoder(frames[4:6], zeros)  → video[16:24],cache updated internally but LOST
```

Each chunk starts with **zero temporal context**. The 3D causal convolutions see no prior frames, causing visible **flickering/discontinuity** at chunk boundaries.

**Advantage**: Only ~300KB transferred per call (just `x`). Fast (~0.5s/chunk).

## 5. Rolling Cache Mode (feat_cache as Explicit I/O)

```python
class DecoderWrapperRolling(nn.Module):
    def forward(self, x, c0, c1, ..., c33):       # 35 inputs
        feat_cache = [c0, c1, ..., c33]
        output = self.decoder(x, feat_cache)
        return tuple([output] + feat_cache)         # 35 outputs (video + updated caches)
```

By making cache tensors **explicit inputs AND outputs**, the NxDModel compilation graph includes the cache update path. The host code can then carry caches between calls:

```python
caches = [zeros] * 34                              # Initialize once
for chunk_idx in range(num_chunks):
    results = decoder_nxd(chunk, *caches)           # Pass old caches IN
    video = results[0]
    caches = results[1:35]                          # Extract updated caches OUT
```

**Behavior**:
```
Chunk 0: decoder(frames[0:2], zeros)    → video[0:8],  cache_0
Chunk 1: decoder(frames[2:4], cache_0)  → video[8:16], cache_1  ← has context!
Chunk 2: decoder(frames[4:6], cache_1)  → video[16:24],cache_2  ← has context!
```

Each chunk receives the **real temporal context** from the previous chunk. The causal convolutions work exactly as intended, producing **flicker-free** video.

**Cost**: ~960MB–1.8GB transferred per call (cache tensors in + out). Slower (~2.2s/chunk) due to host↔device data transfer.

## 6. Comparison Table

| Aspect | diffusers (CPU) | Neuron NoCache | Neuron Rolling |
|--------|----------------|----------------|----------------|
| feat_cache | Mutable list, persists across calls | Registered buffers, always zeros | Explicit I/O, carried between calls |
| Temporal context | Full (21 sequential calls) | None (reset each chunk) | Full (propagated via outputs) |
| Frames per call | 1 | 2 (configurable) | 2 (configurable) |
| Decoder calls | 21 | 11 | 11 |
| Data transfer/call | N/A (CPU) | ~300KB (x only) | ~960MB–1.8GB (x + 34 caches) |
| Video quality | Reference | Flickering at chunk boundaries | Near-reference, flicker-free |

## 7. Temporal Upsample in Rolling Cache

The `upsample3d` layers (at `feat_cache[11]` and `feat_cache[18]`) have special handling. On the **first chunk**, the upsampler sets `feat_cache[idx] = "Rep"` (a sentinel) instead of a tensor. On subsequent chunks, it uses the cached tensor for the `time_conv` (a `WanCausalConv3d` that doubles temporal resolution: t → 2t).

In rolling mode, this works correctly because:
1. First chunk: cache starts as zeros → upsampler initializes normally
2. Subsequent chunks: updated cache tensors carry the temporal state

In NoCache mode, every chunk sees zeros → the upsampler always behaves like the first chunk, losing temporal continuity.

## 8. Cache Sizes by Model

### T2V-A14B (480x832, lh=60, lw=104)

| Cache group | Count | Shape per tensor | Size per tensor | Total |
|-------------|-------|-------------------|-----------------|-------|
| lh×lw (indices 0-11) | 12 | varies×2×60×104 | 2.4–48 MB | ~482 MB |
| 2lh×2lw (indices 12-18) | 7 | varies×2×120×208 | 38–192 MB | ~1,100 MB |
| 4lh×4lw (indices 19-24) | 6 | 192×2×240×416 | 77 MB | ~460 MB |
| 8lh×8lw (indices 25-33) | 9 | varies×2×480×832 | 6–307 MB | ~2,300 MB |
| **Total** | **34** | | | **~1.8 GB/direction** |

### TI2V-5B (512x512, lh=32, lw=32)

| Cache group | Count | Shape per tensor | Size per tensor | Total |
|-------------|-------|-------------------|-----------------|-------|
| lh×lw (indices 0-11) | 12 | varies×2×32×32 | 0.2–4 MB | ~40 MB |
| 2lh×2lw (indices 12-18) | 7 | 1024×2×64×64 | 16 MB | ~112 MB |
| 4lh×4lw (indices 19-24) | 6 | 512×2×128×128 | 32 MB | ~192 MB |
| 8lh×8lw (indices 25-33) | 9 | varies×2×256×256 | 6–64 MB | ~500 MB |
| **Total** | **34** | | | **~960 MB/direction** |

## 9. File Mapping

| File | Purpose |
|------|---------|
| `compile_decoder_v3_rolling.py` / `compile_decoder_rolling.py` | Compilation: wraps decoder with explicit cache I/O |
| `compile_decoder_v3_nocache.py` / `compile_decoder_nocache.py` | Compilation: wraps decoder with zero buffer caches |
| `DecoderWrapperV3Rolling` / `DecoderWrapperV3NoCache` in `neuron_commons.py` | Runtime wrappers for inference |
| `run_*.py` (phase_vae_decode / decoder loading) | Auto-detects rolling vs nocache at inference time |
