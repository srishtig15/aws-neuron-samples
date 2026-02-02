# Qwen-Image-Edit 在 AWS Trainium2 上的部署

中文 | [English](README.md)

本项目实现了在 AWS Trainium2 (trn2) 实例上使用 Neuron SDK 运行 [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) 模型。

## 概述

Qwen-Image-Edit 是一个强大的图像编辑模型，包含以下组件：
- **文本编码器**：Qwen2.5-VL（视觉语言模型）
  - 视觉编码器：32 层 transformer 模块（约 1.4GB）
  - 语言模型：28 层 GQA 架构（约 70 亿参数）
- **Transformer**：QwenImageTransformer2DModel 扩散模型（约 204 亿参数）
- **VAE**：3D 卷积自编码器，用于图像编解码

## 环境要求

- AWS Trainium2 实例（推荐 trn2.48xlarge）
- Neuron SDK 2.x（支持 PyTorch）
- Python 3.10+

```bash
# 激活 Neuron 虚拟环境
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## 环境配置

### 挂载 NVMe 存储（trn2.48xlarge）

trn2.48xlarge 实例配备 4 个 NVMe 设备，需要手动挂载。由于模型文件和编译产物较大，建议挂载到 `/opt/dlami/nvme`：

```bash
# 运行挂载脚本（需要 sudo）
sudo ./setup_nvme.sh
```

该脚本会：
- 自动检测所有可用的 NVMe 设备（排除系统盘）
- 创建 RAID0 阵列以获得最大性能和容量（约 7.5TB）
- 格式化为 ext4 文件系统
- 挂载到 `/opt/dlami/nvme`
- 设置正确的权限（ubuntu 用户可读写）

**注意**：重启后需要重新运行脚本，或添加到 `/etc/fstab` 实现自动挂载。

## 快速开始

### 1. 下载模型

```bash
python neuron_qwen_image_edit/cache_hf_model.py
```

### 2. 编译模型

提供五种编译 API：

| API | 脚本 | 速度 | 说明 |
|-----|------|------|------|
| **V3 CP（最快）** | `compile_transformer_v3_cp.py` | **~0.77s/step** | 上下文并行（TP=4, CP=2）+ NKI Flash Attention |
| V1 Flash | `compile_transformer_v1_flash.py` | ~1.2s/step | parallel_model_trace + NKI Flash Attention |
| V2 Flash | `compile_transformer_v2_flash.py` | ~1.2s/step | ModelBuilder + NKI Flash Attention |
| V2 | `compile_transformer_v2.py` | ~1.2s/step | ModelBuilder API |
| V1 | `compile_transformer.py` | ~2.4s/step | parallel_model_trace API |

**说明**：V3 CP 通过上下文并行将序列分散到 2 个数据并行 rank 上，每个 rank 只处理一半序列同时通过 all-gather 获取完整的 K/V 上下文，从而达到与 H100 相当的性能（~0.77s/step vs H100 的 ~0.75s/step）。

```bash
# 编译 V3 CP（最快，上下文并行 + NKI Flash Attention）
./compile.sh v3_cp

# 或编译 V1 Flash（使用 NKI Flash Attention）
./compile.sh v1_flash

# 或编译 V2 Flash（ModelBuilder + NKI）
./compile.sh v2_flash

# 或编译 V2（ModelBuilder）
./compile.sh v2

# 或只编译 V1
./compile.sh v1
```

默认编译设置：
- 输出图像尺寸：1024x1024
- VAE 分块尺寸：512x512（固定，使用分块处理）
- 最大序列长度：1024
- TP 度：8
- Patch 倍数：3（用于双图合并）

### 3. 运行推理

```bash
# 使用 V3 CP 进行双图合并（最快，上下文并行）
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "将这两个人合成一张婚纱照" \
    --use_v3_cp

# 使用 V1 Flash 进行双图合并（NKI Flash Attention）
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "将这两个人合成一张婚纱照" \
    --use_v1_flash

# 使用 V2 进行双图合并（ModelBuilder）
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "将这两个人合成一张婚纱照" \
    --use_v2

# 使用 V1 进行双图合并
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "将这两个人合成一张婚纱照"

# 单图编辑（需要 patch_multiplier=2）
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "将背景换成海滩" \
    --patch_multiplier 2 \
    --use_v1_flash

# 自定义 CFG 强度
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images input.jpg \
    --prompt "将背景换成海滩" \
    --true_cfg_scale 6.0 \
    --use_v3_cp
```

**注意**：CFG（无分类器引导）在每步中连续运行两次 transformer，而不是 batch_size=2。

## 项目结构

```
hf_pretrained_qwen_image_edit/
├── README.md                      # 英文文档
├── README_CN.md                   # 中文文档（本文件）
├── setup_nvme.sh                  # NVMe RAID0 挂载脚本
├── compile.sh                     # 主编译脚本
├── run_qwen_image_edit.py         # 推理脚本
├── neuron_qwen_image_edit/
│   ├── cache_hf_model.py          # 从 HuggingFace 下载模型
│   ├── compile_vae.py             # VAE 编译
│   ├── compile_transformer.py     # Transformer V1 编译（parallel_model_trace）
│   ├── compile_transformer_v2.py  # Transformer V2 编译（ModelBuilder）
│   ├── compile_transformer_v1_flash.py  # Transformer V1 Flash（NKI Flash Attention）
│   ├── compile_transformer_v2_flash.py  # Transformer V2 Flash（ModelBuilder + NKI）
│   ├── compile_transformer_v3_cp.py     # Transformer V3 CP（上下文并行 + NKI）
│   ├── compile_language_model_v3.py     # 语言模型 V3（TP=4，配合 V3 CP）
│   ├── compile_text_encoder.py    # 文本编码器编译
│   ├── neuron_commons.py          # 通用工具和封装
│   ├── neuron_parallel_utils.py   # 张量并行工具
│   ├── neuron_rope.py             # Neuron 兼容的 RoPE 实现
│   └── autoencoder_kl_qwenimage_neuron.py  # Neuron 兼容的 VAE
├── tests/                         # 单元测试
└── compiled_models/               # 编译模型输出目录
    ├── vae_encoder/
    ├── vae_decoder/
    ├── transformer/               # V1：TP=8 分片 transformer
    ├── transformer_v2/            # V2：ModelBuilder 编译的 transformer
    ├── transformer_v1_flash/      # V1 Flash：NKI Flash Attention
    ├── transformer_v2_flash/      # V2 Flash：ModelBuilder + NKI
    ├── transformer_v3_cp/         # V3 CP：上下文并行（TP=4, CP=2）+ NKI
    ├── language_model_v3/         # 语言模型 V3：TP=4（配合 V3 CP）
    └── vision_encoder/            # 单设备
```

## 技术细节

### 模型架构与执行方式

| 组件 | 总参数量 | 执行方式 | 说明 |
|------|---------|---------|------|
| Transformer（V3 CP） | 204.3 亿 | **TP=4, CP=2** 在 Neuron 上 | 约 10.4 GB/分片，达到 H100 性能水平 |
| Transformer（V1/V2） | 204.3 亿 | TP=8 在 Neuron 上 | 约 5.2 GB/分片 |
| 语言模型（V3） | 70.7 亿 | **TP=4** 在 Neuron 上 | 约 4.1 GB/分片，配合 V3 CP transformer 使用 |
| 语言模型（V1/V2） | 70.7 亿 | CPU | GQA 28Q/4KV 与 TP=8 不兼容 |
| 视觉编码器 | 约 14 亿 | CPU（默认） | 可通过 `--neuron_vision_encoder` 使用 Neuron |
| VAE | 约 3 亿 | DP=8 在 Neuron 上 | 大图使用分块处理 |

**语言模型在 Neuron 上运行（V3）**：使用 V3 CP 时，语言模型可以使用 TP=4 在 Neuron 上运行。这是 GQA 的完美配置：28 个 Q 头 / 4 = 每 rank 7 个头，4 个 KV 头 / 4 = 每 rank 1 个头。使用 `--use_v3_language_model` 配合 `--use_v3_cp`。

**为什么语言模型在 CPU 上运行（V1/V2）**：Qwen2.5-VL 语言模型使用分组查询注意力（GQA），28 个 Q 头和 4 个 KV 头（组大小为 7）。使用 TP=8 时，Q-KV 映射无法正确保持。有效的 TP 度只能是 1、2 或 4。

### 关键技术实现

#### 1. 调制层分片

Transformer 包含 `img_mod` 和 `txt_mod` 调制层。如果不分片，68 亿参数将在每个 TP rank 上重复。

```python
# 每个 block 有调制层 [18432, 3072]
# 60 blocks × 2 mods × 5660 万参数 = 68 亿参数

# 解决方案：使用 ColumnParallelLinear 分片（gather_output=True）
```

**效果**：将 transformer 每分片大小从约 17GB 降到约 5.2GB。

#### 2. Neuron 兼容的 VAE

原始 VAE 使用 `F.interpolate` 的 `mode='nearest-exact'`，Neuron 不支持。自定义 VAE 使用 `mode='nearest'`。

#### 3. Neuron 兼容的 RoPE

原始 RoPE 实现使用复数运算，Neuron 不支持。自定义实现使用实数运算并正确处理交错展开。

#### 4. M-RoPE 位置 ID

Qwen2.5-VL 使用多模态 RoPE，需要 3D position_ids `[3, batch, seq]`：
- 文本 token：顺序位置（t、h、w 维度相同）
- 图像 token：基于空间布局的 3D 网格位置

#### 5. V2 预计算 RoPE

V2 使用 ModelBuilder API，要求将 RoPE 频率作为输入张量传递（而不是在模型内部计算），以避免 XLA 常量折叠问题：

```python
# V1：RoPE 在模型内部计算（与 ModelBuilder 存在 XLA 问题）
# V2/V1 Flash：RoPE 从原始模型预计算并作为输入传递
vid_freqs, txt_freqs = pipe.transformer.pos_embed(video_fhw, max_txt_seq_len=text_seq_len)
img_rotary_emb = torch.stack([vid_freqs.real, vid_freqs.imag], dim=-1)  # [num_patches, 64, 2]
txt_rotary_emb = torch.stack([txt_freqs.real, txt_freqs.imag], dim=-1)  # [text_seq, 64, 2]
```

RoPE 在编译时缓存，推理时加载。

#### 6. V3 CP 上下文并行

V3 CP 通过结合上下文并行与张量并行达到 H100 水平的性能：

**架构**：TP=4, CP=2（world_size=8）
- **张量并行（TP=4）**：将模型权重分片到 4 个设备
- **上下文并行（CP=2）**：每个 CP rank 处理一半序列

**关键实现细节**：

1. **SPMDRank 运行时 Rank 检测**：使用 `neuronx_distributed.parallel_layers.layers.SPMDRank` 在运行时（而非 trace 时）获取正确的全局 rank。这对于正确的 scatter/gather 操作至关重要。

```python
from neuronx_distributed.parallel_layers.layers import SPMDRank

class NeuronQwenTransformerV3CP(nn.Module):
    def __init__(self, ...):
        self.global_rank = SPMDRank(world_size=world_size)
        self.data_parallel_group = parallel_state.get_data_parallel_group()

    def forward(self, hidden_states, ...):
        # 在运行时计算 DP rank
        dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.tp_degree)
        # 基于 DP rank 分散输入
        hidden_states = scatter_to_process_group_spmd(hidden_states, dim=1, rank=dp_rank, ...)
```

2. **K/V All-Gather**：每个 CP rank 从所有 CP rank 收集完整的 K/V 以看到完整上下文，同时只计算其本地 query 部分的注意力。

```python
# 在注意力模块中
if self.context_parallel_enabled:
    # 跨 CP 组收集完整的 K/V
    key = gather_from_tensor_model_parallel_region_with_dim(
        key, dim=2, process_group=self.data_parallel_group
    )
    value = gather_from_tensor_model_parallel_region_with_dim(
        value, dim=2, process_group=self.data_parallel_group
    )
```

3. **输出 All-Gather**：注意力计算后，输出通过 all-gather 重建完整序列。

**性能提升**：相比 V1 Flash 约 1.56 倍加速（0.77s/step vs 1.2s/step），原因：
- 每个 rank 只处理 50% 的 query 序列
- 通信开销被 NKI Flash Attention 的效率所摊销

#### 7. 语言模型 V3 在 Neuron 上运行

使用 V3 CP transformer（TP=4）时，语言模型也可以使用 TP=4 在 Neuron 上运行，而不是在 CPU 上：

**为什么 TP=4 是语言模型的完美配置**：
- Q 头：28 / 4 = 每 rank 7 个头（整除）
- KV 头：4 / 4 = 每 rank 1 个头（整除）
- 无需填充或复制！

**编译**：语言模型使用 ModelBuilder API 编译，与 V3 CP transformer 使用相同的 `world_size=8`：

```bash
# 编译 V3 CP（包含语言模型 V3）
./compile.sh v3_cp
```

**使用方法**：
```bash
NEURON_RT_NUM_CORES=8 python run_qwen_image_edit.py \
    --images img1.png img2.png \
    --prompt "将这两个人合成" \
    --use_v3_cp \
    --use_v3_language_model
```

**注意**：语言模型 V3 需要 V3 CP transformer（两者都使用 world_size=8）。使用 V1/V2/V1 Flash/V2 Flash（TP=8）时，语言模型必须在 CPU 上运行。

#### 8. V1 Flash 与 NKI Flash Attention

V1 Flash 结合了两种方法的优点：
- 使用 `parallel_model_trace` API（与 V1 相同），支持 NKI 内核
- 使用预计算 RoPE（与 V2 相同），避免 XLA 问题
- 使用 NKI Flash Attention 内核进行硬件优化的注意力计算

**关键实现细节**：

1. **禁用 XLA 函数化**：NKI 内核需要原地输出修改，与 XLA 的函数化冲突。设置 `XLA_DISABLE_FUNCTIONALIZATION=1` 至关重要。

2. **自定义注意力模块**：用 `NKIQwenAttention` 替换 diffusers 的 `Attention` 类，直接调用 NKI 内核：

```python
# NKI Flash Attention 封装（与 Flux 实现相同）
def nki_flash_attention(query, key, value):
    # 输入：[B, H, S, D]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16)
    _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    return attn_output.reshape((bs, n_head, q_len, d_head))
```

3. **NKI 内核导入**：使用与 Flux 相同的导入路径以保持兼容：

```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
_flash_fwd_call = nki_jit()(attention_isa_kernel)
```

## 编译选项

```bash
./compile.sh [HEIGHT] [WIDTH] [IMAGE_SIZE] [TP_DEGREE] [MAX_SEQ_LEN] [PATCH_MULTIPLIER]

# 示例：
./compile.sh 1024 1024 448 8 1024 3   # 默认：1024x1024，双图合并
./compile.sh 1024 1024 448 8 1024 2   # 单图编辑
./compile.sh 512 512 224 8 512 2      # 512x512 输出
```

## 推理选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--images` | 必需 | 输入图像路径，支持 1-3 张图 |
| `--prompt` | 必需 | 编辑指令 |
| `--use_v3_cp` | False | 使用 V3 CP transformer（上下文并行 + NKI，最快） |
| `--use_v3_language_model` | False | 使用 V3 语言模型在 Neuron 上运行（需要 V3 CP） |
| `--use_v1_flash` | False | 使用 V1 Flash transformer（NKI Flash Attention） |
| `--use_v2_flash` | False | 使用 V2 Flash transformer（ModelBuilder + NKI） |
| `--use_v2` | False | 使用 V2 transformer（ModelBuilder） |
| `--height` | 1024 | 输出图像高度（必须与编译模型匹配） |
| `--width` | 1024 | 输出图像宽度（必须与编译模型匹配） |
| `--patch_multiplier` | 2 | 2=单图编辑，3=双图合并 |
| `--image_size` | 448 | 视觉编码器输入尺寸 |
| `--max_sequence_length` | 1024 | 文本序列长度（必须与编译匹配） |
| `--num_inference_steps` | 40 | 去噪步数 |
| `--true_cfg_scale` | 4.0 | CFG 强度（每步运行两次 transformer） |
| `--cpu_vision_encoder` | True | 在 CPU 上运行视觉编码器（默认） |
| `--neuron_vision_encoder` | False | 在 Neuron 上运行视觉编码器 |
| `--vae_tile_size` | 512 | VAE 分块处理尺寸 |

## 故障排除

### 形状不匹配错误
```
ERROR: hidden_states has X patches but model expects Y
```
**解决方案**：确保 `--height`、`--width` 和 `--patch_multiplier` 与编译时使用的维度匹配。

### 内存不足错误
```
Failed to allocate X bytes
```
**解决方案**：
1. 确保调制层已分片
2. 所有并行层使用 `dtype=torch.bfloat16`
3. 减小图像尺寸或序列长度

### RoPE 维度不匹配
```
Shapes are not compatible: f32[1,2048,3,128] vs f32[1,1024,1,128]
```
**解决方案**：确保编译和推理之间的 `patch_multiplier` 匹配。

### 检查点文件过大（master_weight 问题）

编译后，如果检查点文件比预期大约 2 倍（例如 transformer 80GB 而不是 40GB，语言模型 28GB 而不是 14GB），说明 `shard_checkpoint()` 函数同时保存了 `master_weight`（完整未分片）和 `weight`（已分片）张量。

**解决方案**：从检查点中删除 `master_weight` 张量：

```python
from safetensors.torch import load_file, save_file
import os

def cleanup_checkpoint(weights_dir):
    """从分片检查点中删除 master_weight 张量"""
    for rank in range(4):
        path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors")
        data = dict(load_file(path))
        # 删除 master_weight 张量
        cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}
        save_file(cleaned, path)
        print(f"tp{rank}: {len(data)} -> {len(cleaned)} tensors")

# 清理 transformer V3 CP
cleanup_checkpoint("/path/to/compiled_models/transformer_v3_cp/weights")

# 清理语言模型 V3
cleanup_checkpoint("/path/to/compiled_models/language_model_v3/weights")
```

这可以将检查点大小减少约 50%，且不影响功能。

### 缺少 inv_freq 张量（语言模型 V3）

```
RuntimeError: Missing weight tensor with key language_model.language_model.rotary_emb.inv_freq
```

**原因**：旋转位置编码的 `inv_freq` 缓冲区默认不包含在 `state_dict()` 中。

**解决方案**：从原始模型添加 `inv_freq` 缓冲区到检查点：

```python
from safetensors.torch import load_file, save_file
from diffusers import QwenImageEditPlusPipeline
import torch

# 加载原始模型
pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
lm = pipe.text_encoder.model.language_model

# 收集 inv_freq 缓冲区
inv_freq_buffers = {}
for name, buf in lm.named_buffers():
    if 'inv_freq' in name:
        full_key = f"language_model.language_model.{name}"
        inv_freq_buffers[full_key] = buf.to(torch.bfloat16).clone()

# 添加到所有 TP 检查点
for rank in range(4):
    path = f"weights/tp{rank}_sharded_checkpoint.safetensors"
    data = dict(load_file(path))
    data.update(inv_freq_buffers)
    save_file(data, path)
```

## 性能

### 基准测试结果（1024x1024，双图合并，40 步 + CFG）

| 平台 | Transformer 速度 | 总时间 |
|------|-----------------|--------|
| H100（无 Flash Attention） | ~0.75s/step | ~60s |
| **TRN2 V3 CP（上下文并行 + NKI）** | **~0.77s/step** | **~62s** |
| TRN2 V1 Flash（NKI） | ~1.2s/step | ~96s |
| TRN2 V2 Flash（ModelBuilder + NKI） | ~1.2s/step | ~96s |
| TRN2 V2（ModelBuilder） | ~1.2s/step | ~96s |
| TRN2 V1（parallel_model_trace） | ~2.4s/step | ~190s |

**V3 CP 达到 H100 水平的性能！** 通过使用上下文并行（CP=2）结合张量并行（TP=4），每个 rank 只处理一半序列，相比 V1 Flash 实现约 1.56 倍加速。

### 性能分析

#### 为什么 V1 → V2 实现了 2 倍加速？

关键区别在于 **RoPE（旋转位置编码）处理方式**：

| 版本 | RoPE 处理 | 开销 |
|------|----------|------|
| V1 | 每次前向传播在模型内部计算 | 高 - 复数运算、位置编码 |
| V2/V1 Flash/V2 Flash | 预计算一次，作为输入传递 | 低 - 仅张量传递 |

**预计算 RoPE 是主要的性能提升**（2 倍加速），而非编译 API 或注意力实现。

#### 为什么 V2 ≈ V1 Flash ≈ V2 Flash？

| 版本 | RoPE | 注意力 | 速度 |
|------|------|--------|------|
| V1 | 模型内部 | 标准 SDPA | ~2.4s/step |
| V2 | **预计算** | 标准 SDPA | ~1.2s/step |
| V1 Flash | **预计算** | NKI Flash | ~1.2s/step |
| V2 Flash | **预计算** | NKI Flash | ~1.2s/step |

**关键发现**：
1. **预计算 RoPE** 是主要优化（V1 → V2：2 倍加速）
2. **NKI Flash Attention vs 编译器优化的 SDPA**：无显著差异 - Neuron 编译器已经很好地优化了 SDPA
3. **编译 API（parallel_model_trace vs ModelBuilder）**：对最终性能影响很小

### V1 vs V2 vs V1 Flash vs V2 Flash vs V3 CP 对比

| 方面 | V1 | V2 | V1 Flash | V2 Flash | **V3 CP（最快）** |
|------|-----|-----|----------|----------|-------------------|
| 编译 API | `parallel_model_trace` | `ModelBuilder` | `parallel_model_trace` | `ModelBuilder` | `ModelBuilder` |
| 注意力 | 标准 SDPA | 标准 SDPA | NKI Flash | NKI Flash | **NKI Flash** |
| RoPE 处理 | 模型内部 | 预计算 | 预计算 | 预计算 | 预计算 |
| 并行方式 | TP=8 | TP=8 | TP=8 | TP=8 | **TP=4, CP=2** |
| 速度 | ~2.4s/step | ~1.2s/step | ~1.2s/step | ~1.2s/step | **~0.77s/step** |
| 核心优势 | 简单 | 预计算 RoPE | NKI | 两者兼具 | **上下文并行** |

**推荐**：使用 **V3 CP** 获得最快性能（达到 H100 水平）。如果需要更简单的调试或不想使用上下文并行的复杂性，可以使用 V1 Flash。

### 编译器优化

以下编译器标志用于获得最佳性能：

```python
--model-type=transformer      # Transformer 专用优化
-O1                           # 优化级别
--auto-cast=none              # 保持 bfloat16 精度
--tensorizer-options='--enable-ccop-compute-overlap'  # 通信与计算重叠
```

`--enable-ccop-compute-overlap` 标志对张量并行特别重要，它允许 all-reduce 操作与计算重叠，减少同步开销。

### 组件耗时分解

#### V3 CP 配合 V3 语言模型（最快）

| 组件 | 时间 |
|------|------|
| 视觉编码器（CPU，首次调用） | ~21s |
| 视觉编码器（CPU，缓存后） | ~0.7s |
| **语言模型（Neuron V3）** | **~0.5s** |
| VAE 编码 | ~0.4s |
| **Transformer V3 CP（40 步 × 2 CFG）** | **~62s** |
| VAE 解码 | ~1.6s |

#### V1 Flash / V2 配合 CPU 语言模型

| 组件 | 时间 |
|------|------|
| 视觉编码器（CPU，首次调用） | ~21s |
| 视觉编码器（CPU，缓存后） | ~0.7s |
| 语言模型（CPU） | ~1-4s |
| VAE 编码 | ~0.4s |
| **Transformer（40 步 × 2 CFG）** | **~96s** |
| VAE 解码 | ~1.6s |

Transformer 是主要瓶颈，占总推理时间的约 80%。V3 语言模型在 Neuron 上比 CPU 快约 2-8 倍。

## 已知限制

1. **固定尺寸**：模型针对特定尺寸编译，不同尺寸需要重新编译。
2. **语言模型（V1/V2）**：由于 GQA 架构（28Q/4KV 头与 TP=8 不兼容），在 CPU 上运行。使用 V3 CP 配合 V3 语言模型可在 Neuron 上运行。
3. **序列长度**：编译和推理之间必须匹配。
4. **NKI Flash Attention**：需要 `XLA_DISABLE_FUNCTIONALIZATION=1` 才能在 parallel_model_trace（V1 Flash）和 ModelBuilder（V2 Flash）中工作。没有此标志，NKI 内核会报"不可变输出参数"错误。
5. **语言模型 V3**：仅与 V3 CP transformer 兼容（两者都使用 world_size=8，TP=4）。

## 参考资料

- [Qwen-Image-Edit 模型](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [AWS Neuron SDK 文档](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Diffusers 库](https://github.com/huggingface/diffusers)

## 许可证

本代码仅供研究和教育目的使用。使用条款请参考原始模型许可证。
