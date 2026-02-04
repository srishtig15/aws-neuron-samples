# Wan2.2 Text-to-Video on AWS Trainium2 - V3 Flash Implementation

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

This implementation runs **Wan2.2-TI2V-5B** (Text-Image-to-Video) model on AWS Trainium2 (trn2.48xlarge) using the NxD Inference framework with NKI Flash Attention optimization.

**Key Features:**
- TP=8 tensor parallelism for transformer
- NKI Flash Attention for efficient attention computation
- Optimized VAE decoder with batched frame processing
- Support for 512x512 resolution, 81 frames output

### Architecture

| Component | Implementation | Details |
|-----------|---------------|---------|
| Text Encoder | UMT5 with TP=8 | NxDModel, weights sharded across 8 ranks |
| Transformer | DiT with TP=8 | NKI Flash Attention, ~40s for 50 steps |
| VAE Decoder | TorchScript (V1) | Batched processing (2 frames/call), ~24s |
| post_quant_conv | TorchScript (V1) | DataParallel on 4 cores |

### Why V1 Decoder?

We use the V1 (TorchScript) decoder instead of V2 (NxDModel) for the following reasons:

| Aspect | V1 (TorchScript) | V2 (NxDModel) |
|--------|------------------|---------------|
| Loading | `torch.jit.load()` | `NxDModel.load()` with world_size=8 |
| Weights | Single copy | 8 copies (duplicated to all ranks) |
| Framework overhead | Minimal | NxD process group synchronization |
| **Decoder time** | **~24s** | **~60s** |

The V2 decoder runs within NxDParallelState(world_size=8) but doesn't actually use tensor parallelism (weights are duplicated, not sharded). This creates unnecessary overhead without parallelization benefits.

### Key Optimizations

#### 1. NKI Flash Attention (Transformer)

The transformer uses NKI (Neuron Kernel Interface) Flash Attention for efficient self-attention computation:

```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)
```

This avoids materializing the full attention matrix, reducing memory usage and improving speed.

#### 2. Batched Decoder Processing

Original implementation calls decoder 21 times (once per latent frame):
```python
# Original: 21 calls, ~60s total
for i in range(num_frame):  # num_frame=21
    out_ = self.decoder(x[:, :, i:i+1, :, :], ...)
```

Optimized implementation processes 2 frames per call:
```python
# Optimized: 11 calls, ~24s total
i = 0
while i < num_frame:
    if i == 0:
        out = self.decoder(x[:, :, 0:1, :, :], first_chunk=True)
        i += 1
    else:
        out_ = self.decoder(x[:, :, i:i+2, :, :], ...)  # 2 frames
        i += 2
```

This reduces Neuron kernel launch overhead by ~50%.

### Installation

```bash
# Activate the Neuron venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Clone the repository (if not already done)
cd /home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_ti2v

# Set PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Compilation

Run the compilation script (takes ~30-60 minutes):

```bash
./compile_v3_flash.sh /opt/dlami/nvme/compiled_models_v3_flash
```

This compiles:
1. Text Encoder (V2, TP=8)
2. Transformer (V3 Flash, TP=8, NKI)
3. Decoder + post_quant_conv (V1, TorchScript)

### Inference

```bash
NEURON_RT_NUM_CORES=8 python run_wan2.2_ti2v_v3_flash.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_flash \
    --prompt "A cat walks on the grass, realistic" \
    --force_v1_decoder
```

**Parameters:**
- `--compiled_models_dir`: Path to compiled models
- `--prompt`: Text prompt for video generation
- `--image_path`: (Optional) Input image for image-to-video
- `--seed`: Random seed (default: 42)
- `--num_inference_steps`: Diffusion steps (default: 50)
- `--force_v1_decoder`: Use V1 decoder (recommended)

### Performance

On trn2.48xlarge (8 Neuron cores):

| Metric | Value |
|--------|-------|
| Total inference time | ~64s |
| Transformer (50 steps) | ~40s (0.8s/step) |
| VAE Decoder | ~24s |
| Output | 84 frames @ 512x512 |

### File Structure

```
hf_pretrained_wan2.2_ti2v/
├── compile_v3_flash.sh              # Compilation script
├── run_wan2.2_ti2v_v3_flash.py      # Inference script
├── autoencoder_kl_wan.py            # Modified VAE with batched decoding
└── neuron_wan2_2_ti2v/
    ├── compile_text_encoder_v2.py   # Text encoder compilation
    ├── compile_transformer_v3_flash.py  # Transformer with NKI
    ├── compile_decoder.py           # V1 decoder (TorchScript)
    ├── compile_decoder_v2_optimized.py  # V2 decoder (NxDModel)
    └── neuron_commons.py            # Wrapper classes
```

---

<a name="中文"></a>
## 中文

### 概述

本实现在 AWS Trainium2 (trn2.48xlarge) 上运行 **Wan2.2-TI2V-5B**（文本/图像到视频）模型，使用 NxD Inference 框架和 NKI Flash Attention 优化。

**主要特性：**
- Transformer 使用 TP=8 张量并行
- NKI Flash Attention 高效注意力计算
- VAE Decoder 批量帧处理优化
- 支持 512x512 分辨率，81 帧输出

### 架构

| 组件 | 实现方式 | 详情 |
|------|---------|------|
| Text Encoder | UMT5, TP=8 | NxDModel，权重分片到 8 个 rank |
| Transformer | DiT, TP=8 | NKI Flash Attention，50 步约 40s |
| VAE Decoder | TorchScript (V1) | 批量处理（每次 2 帧），约 24s |
| post_quant_conv | TorchScript (V1) | DataParallel 在 4 核上运行 |

### 为什么使用 V1 Decoder？

我们使用 V1 (TorchScript) decoder 而不是 V2 (NxDModel)，原因如下：

| 方面 | V1 (TorchScript) | V2 (NxDModel) |
|------|------------------|---------------|
| 加载方式 | `torch.jit.load()` | `NxDModel.load()`, world_size=8 |
| 权重内存 | 单份 | 8 份（复制到所有 rank）|
| 框架开销 | 最小 | NxD 进程组同步 |
| **Decoder 耗时** | **约 24s** | **约 60s** |

V2 decoder 在 NxDParallelState(world_size=8) 内运行，但实际不使用张量并行（权重是复制的，不是分片的）。这带来了不必要的开销，却没有并行化收益。

### 关键优化

#### 1. NKI Flash Attention（Transformer）

Transformer 使用 NKI（Neuron Kernel Interface）Flash Attention 进行高效的自注意力计算：

```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)
```

这避免了实例化完整的注意力矩阵，减少内存使用并提高速度。

#### 2. 批量 Decoder 处理

原始实现每次调用 decoder 处理 1 帧（共 21 次）：
```python
# 原始：21 次调用，总计约 60s
for i in range(num_frame):  # num_frame=21
    out_ = self.decoder(x[:, :, i:i+1, :, :], ...)
```

优化后每次处理 2 帧：
```python
# 优化后：11 次调用，总计约 24s
i = 0
while i < num_frame:
    if i == 0:
        out = self.decoder(x[:, :, 0:1, :, :], first_chunk=True)
        i += 1
    else:
        out_ = self.decoder(x[:, :, i:i+2, :, :], ...)  # 2 帧
        i += 2
```

这将 Neuron kernel launch 开销减少了约 50%。

### 安装

```bash
# 激活 Neuron 虚拟环境
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 进入项目目录
cd /home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_ti2v

# 设置 PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### 编译

运行编译脚本（约需 30-60 分钟）：

```bash
./compile_v3_flash.sh /opt/dlami/nvme/compiled_models_v3_flash
```

编译内容：
1. Text Encoder (V2, TP=8)
2. Transformer (V3 Flash, TP=8, NKI)
3. Decoder + post_quant_conv (V1, TorchScript)

### 推理

```bash
NEURON_RT_NUM_CORES=8 python run_wan2.2_ti2v_v3_flash.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_flash \
    --prompt "一只猫在草地上行走，真实感" \
    --force_v1_decoder
```

**参数说明：**
- `--compiled_models_dir`: 编译模型路径
- `--prompt`: 视频生成的文本提示
- `--image_path`: （可选）输入图像，用于图生视频
- `--seed`: 随机种子（默认：42）
- `--num_inference_steps`: 扩散步数（默认：50）
- `--force_v1_decoder`: 使用 V1 decoder（推荐）

### 性能

在 trn2.48xlarge（8 个 Neuron 核心）上：

| 指标 | 数值 |
|------|------|
| 总推理时间 | 约 64s |
| Transformer（50 步）| 约 40s（0.8s/步）|
| VAE Decoder | 约 24s |
| 输出 | 84 帧 @ 512x512 |

### 文件结构

```
hf_pretrained_wan2.2_ti2v/
├── compile_v3_flash.sh              # 编译脚本
├── run_wan2.2_ti2v_v3_flash.py      # 推理脚本
├── autoencoder_kl_wan.py            # 修改后的 VAE（批量解码）
└── neuron_wan2_2_ti2v/
    ├── compile_text_encoder_v2.py   # Text encoder 编译
    ├── compile_transformer_v3_flash.py  # Transformer (NKI)
    ├── compile_decoder.py           # V1 decoder (TorchScript)
    ├── compile_decoder_v2_optimized.py  # V2 decoder (NxDModel)
    └── neuron_commons.py            # Wrapper 类
```

### 版本历史

| 版本 | 说明 |
|------|------|
| V1 | 基础实现，torch_neuronx.trace |
| V2 | NxDModel/ModelBuilder API |
| V3 CP | Context Parallel（有 collective ops 序列化问题）|
| **V3 Flash** | NKI Flash Attention，当前推荐版本 |

---

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- [Wan-AI/Wan2.2](https://github.com/Wan-Video/Wan2.1) - Original model
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/) - Trainium/Inferentia support
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Pipeline implementation
