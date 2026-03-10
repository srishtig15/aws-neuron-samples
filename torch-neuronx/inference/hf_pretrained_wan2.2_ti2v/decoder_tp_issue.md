# Wan2.2 VAE Decoder on Trainium2: TP Sharding Causes Blurry Output, No-TP Exceeds Instruction Limit

## Context

We're running Wan2.2-TI2V-5B video generation on Trainium2 (trn2.48xlarge). The transformer (5B params) uses TP=4 + CP=2 across 8 NeuronCores and works correctly. The problem is the **VAE decoder**.

## The Problem

At 720P resolution (1280x704, 81 frames), the VAE decoder cannot be compiled without tensor parallelism — it generates **16.7M instructions**, far exceeding the 5M limit. So we must use TP to reduce per-core instruction count.

However, TP sharding of the decoder produces **visibly blurry/degraded video output**, while the CPU decoder produces correct quality. We've confirmed this through A/B testing with identical latents.

## Decoder Architecture

The Wan2.2 VAE decoder is a Conv3D-based network:

```
conv_in(48 -> 1024) -> mid_block(1024) -> up_block_0(1024 -> 1024, upsample3d)
-> up_block_1(1024 -> 1024, upsample3d) -> up_block_2(1024 -> 512, upsample2d)
-> up_block_3(512 -> 256) -> conv_out(256 -> 12)
```

We use rolling cache with `decoder_frames=2` to process latent frames in chunks.

## What We've Tried

All TP approaches produce blurry output or fail to run:

### 1. Group Conv (no communication) — Blurry

Each rank computes `Conv3d(C/tp, O/tp)` independently — only sees 1/tp of input channels. At TP=4, this loses **75% of cross-channel weight connections**. Fast and fits in memory, but mathematically approximate.

**Result: Visibly blurry output.**

### 2. Allgather (exact for early blocks) — OOM on later blocks

`gather(x_sharded) -> Conv3d(C_full, O/tp)` — mathematically exact. Works for blocks 0-1 (small spatial dims, ~231MB allgather buffer). But blocks 2-3 at 720P need **3.7-7.4 GB** allgather buffers.

**Result: OOM on blocks 2-3.**

### 3. Reduce-Scatter (exact) — OOM at NEFF load

`Conv3d(C/tp, O_full) -> reduce_scatter(output)` — mathematically exact. But the full-output intermediate tensors make the NEFF too large to coexist with the transformer on the same device.

**Result: OOM at NEFF load time** (`Failed to allocate 922MB buffer`).

### 4. Hybrid: Allgather (blocks 0-1) + Group Conv (blocks 2-3) — Still Blurry

Exact computation for early high-channel blocks via allgather, approximate group conv for later blocks only. 15 out of ~44 conv layers use group conv.

**Result: Still visibly blurry.**

## Memory Constraint

The transformer occupies ~20 GB of 22.6 GB HBM per NC pair. The decoder must fit in the remaining **~2-3 GB** (NEFF code + weights + scratchpad).

## Official Wan2.2 Reference

The official Wan2.2 repository ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)) does **not** shard the VAE decoder at all, even with 8 GPUs. It only decodes on rank 0:

```python
# wan/text2video.py
if self.rank == 0:
    videos = self.vae.decode(x0)
```

Their multi-GPU strategy (FSDP + DeepSpeed Ulysses) only applies to the DiT transformer and T5 text encoder.

## Questions for the Neuron Team

1. **Graph partitioning**: Can the compiler automatically split a single model into multiple sub-graphs (pipeline stages) that each fit within the 5M instruction limit? Or is there a supported way to do this manually with NxD?

2. **Instruction limit**: Is there a way to raise or work around the 5M instruction limit for large Conv3D models?

3. **Memory-efficient exact TP**: Is there a fused allgather+conv primitive that avoids materializing the full gathered tensor in scratchpad? (e.g., streaming allgather where each chunk is consumed by the conv immediately)

4. **Any other suggestions** for compiling and running a 16.7M-instruction Conv3D decoder on Neuron at 720P with correct numerical results?

## Environment

| Component | Version / Config |
|-----------|-----------------|
| Instance | trn2.48xlarge (16 NeuronCores / 8 NC pairs) |
| Neuron SDK | PyTorch 2.9 + NxD Inference |
| Model | Wan2.2-TI2V-5B (HuggingFace Diffusers) |
| Resolution | 1280x704, 81 frames, decoder_frames=2 |
| TP config | TP=4 for decoder (transformer uses TP=4, CP=2) |
| Compiler flags | `--target=trn2 --lnc=2 --model-type=unet-inference -O1 --auto-cast=none` |
