# Qwen-Image-Edit Neuron 修复总结

## 问题描述
推理输出图像是乱码/模糊的，尽管单元测试显示各个组件（VAE、Transformer）都通过了。

## 根本原因分析

经过详细调试，发现了 **两个关键问题**：

### 问题 1: Processor 输出尺寸与编译的 Vision Encoder 不匹配

**原因**：
- Vision Encoder 编译时使用 `image_size=224`，期望输入 256 个 patches (16×16)
- 但是 Qwen2.5-VL 的 processor 会根据输入图像动态调整输出尺寸
- 当输入 512×512 图像时，processor 输出 1296 个 patches (36×36)
- 这导致 Vision Encoder 无法正确处理输入

**证据**：
```python
# 512×512 图像
pixel_values: torch.Size([1296, 1176])  # 36×36 patches
image_grid_thw: [[1, 36, 36]]

# 224×224 图像 (正确)
pixel_values: torch.Size([256, 1176])   # 16×16 patches
image_grid_thw: [[1, 16, 16]]
```

### 问题 2: 多模态输入的 Position IDs 计算错误

**原因**：
- Qwen2.5-VL 使用 **M-RoPE (Multimodal Rotary Position Embeddings)**
- 需要 3D position_ids，形状为 `[3, batch_size, seq_len]`
- 对于文本 token：三个维度 (t, h, w) 都是相同的顺序位置
- 对于图像 token：需要根据图像的空间布局计算 3D 网格位置
- 原代码只实现了简单的文本顺序位置，没有处理图像的 3D 位置

**证据**：
```python
# 修复前 (文本-only 测试)
cpu_lm: Cosine Sim = 1.000175  # 通过

# 修复前 (带图像测试 - 未测试)
# 实际推理失败，因为 position_ids 计算错误
```

## 修复方案

### 修复 1: 配置 Processor 输出固定尺寸

**文件**: `run_qwen_image_edit.py`

**修改** (第 822-829 行):
```python
# CRITICAL: Configure processor to output fixed image size matching compiled vision encoder
# The processor dynamically determines grid size based on min/max pixels.
# We must force it to use the exact size the vision encoder was compiled for.
target_pixels = args.image_size * args.image_size
print(f"\nConfiguring processor for vision encoder size: {args.image_size}x{args.image_size}")
print(f"  Setting min_pixels = max_pixels = {target_pixels}")
pipe.processor.image_processor.min_pixels = target_pixels
pipe.processor.image_processor.max_pixels = target_pixels
```

### 修复 2: 实现正确的 M-RoPE Position IDs 计算

**文件**: `neuron_qwen_image_edit/neuron_commons.py`

**新增方法** `_get_rope_index()`:
```python
def _get_rope_index(self, input_ids, image_grid_thw, attention_mask):
    """
    Calculate 3D position_ids for M-RoPE (Multimodal RoPE).

    For multimodal input (text + images), position_ids have different patterns:
    - Text tokens: sequential positions (same for t, h, w dimensions)
    - Image tokens: 3D grid positions based on spatial layout
    """
    # ... 完整实现见代码
```

**更新 forward() 方法**:
```python
# Step 4: Calculate 3D position_ids for M-RoPE (required by Qwen2.5-VL)
position_ids = self._get_rope_index(input_ids, image_grid_thw, attention_mask)
```

### 修复 3: Position IDs 长度不匹配 Bug

**问题**：
首次实现的 `_get_rope_index()` 使用 `image_grid_thw` 计算图像 token 数量，但这与 `input_ids` 中实际的图像 token 数量不一致，导致：
```
RuntimeError: shape mismatch: value tensor of shape [3, 134] cannot be broadcast
to indexing result of shape [3, 129]
```

**修复**：
重写 `_get_rope_index()` 方法，逐 token 处理，直接使用 `input_ids` 中的实际图像 token 数量：
```python
# 逐 token 处理，确保输出长度与输入匹配
for i in range(valid_len):
    if is_image_token[i]:
        # Image token: use 2D grid position
        grid_idx = img_token_idx
        h_pos = (grid_idx % (llm_grid_h * llm_grid_w)) // llm_grid_w
        w_pos = (grid_idx % (llm_grid_h * llm_grid_w)) % llm_grid_w
        pos_list.append([current_pos + t_pos, current_pos + h_pos, current_pos + w_pos])
        img_token_idx += 1
    else:
        # Text token: use sequential position
        pos_list.append([current_pos, current_pos, current_pos])
        current_pos += 1
```

### 修复 4: CPU Vision Encoder 选项 (精度优化)

**问题**：
详细组件测试发现，编译后的 Vision Encoder 存在精度损失 (cosine=0.997)，这个差异通过 28 层 Language Model 被放大：

```
详细组件测试结果:
  Vision Encoder: cosine=0.997474 (有差异)
  Position IDs: 完全匹配 ✓
  LM (Vision差异隔离): cosine=0.988462 (差异被放大)
  LM (Position差异隔离): cosine=0.999971 (几乎完美)
```

**修复**：
添加 `--cpu_vision_encoder` 选项，在 CPU 上运行 Vision Encoder 以获得最高精度：

**文件**: `run_qwen_image_edit.py`
```python
parser.add_argument("--cpu_vision_encoder", action="store_true",
                    help="Run Vision Encoder on CPU for higher accuracy. "
                         "Use this if output images are blurry/corrupted.")
```

**文件**: `neuron_qwen_image_edit/neuron_commons.py`
```python
# NeuronTextEncoderWrapper.__init__
self.cpu_vision_encoder = cpu_vision_encoder
self.use_cpu_vision_encoder = cpu_vision_encoder is not None

# forward() 方法
if self.use_cpu_vision_encoder:
    with torch.no_grad():
        image_embeds = self.cpu_vision_encoder(pixel_values, image_grid_thw)
```

**使用方法**：
```bash
python run_qwen_image_edit.py \
    --images image1.png \
    --prompt "把女生变成男生" \
    --cpu_vision_encoder  # 使用 CPU Vision Encoder 获得更高精度
```

## 测试结果

### 修复后的单元测试

| 测试 | 结果 | Cosine Similarity |
|------|------|-------------------|
| VAE | PASS | 0.999+ |
| Transformer | PASS | 0.998+ |
| Text Encoder (text-only) | PASS | 1.000175 |
| Text Encoder (multimodal) | PASS | 0.996193 |

### 多模态测试
```
============================================================
RESULTS (Multimodal Text + Image)
============================================================
  Cosine Similarity: 0.996193
  Max Absolute Error: 4.500000e+01
  [PASS] Multimodal text encoder works correctly!
```

## 运行推理

确保使用正确的参数运行推理：

```bash
# 确保 image_size 与编译时一致
python run_qwen_image_edit.py \
    --images image1.png \
    --prompt "把女生变成男生" \
    --image_size 224 \
    --compiled_height 512 \
    --compiled_width 512 \
    --transformer_batch_size 1
```

**重要参数**:
- `--image_size 224`: Vision Encoder 编译尺寸，必须与 compile.sh 中的 IMAGE_SIZE 一致
- `--compiled_height/width 512`: Transformer 和 VAE 编译尺寸

## 文件修改列表

1. `run_qwen_image_edit.py`:
   - 添加 processor 配置代码
   - 添加 `--cpu_vision_encoder` 参数 (现在是默认)
   - 添加 `--neuron_vision_encoder` 参数 (覆盖默认)
   - 在 `load_all_compiled_models()` 中添加 CPU Vision Encoder 支持
   - 添加 VAE_IMAGE_SIZE 覆盖，确保 patch 数量匹配编译配置

2. `neuron_qwen_image_edit/neuron_commons.py`:
   - 添加 `image_token_id` 和 `vision_start_token_id` 属性
   - 添加 `_get_rope_index()` 方法 - 逐 token 处理版本
   - 更新 `forward()` 方法使用新的 position_ids 计算
   - 添加 `cpu_vision_encoder` 参数和使用逻辑

3. `neuron_qwen_image_edit/neuron_rope.py`:
   - 修复 `NeuronQwenEmbedRope.forward()` 的 img_shapes 解析逻辑
   - 修复 `apply_rotary_emb_neuron()` 的 cos/sin 扩展方式 (repeat -> repeat_interleave)
   - 修复 x_rotated 的构建方式

4. `tests/test_multimodal.py` (新增):
   - 多模态测试，验证图像+文本处理

5. `test_rope_comparison.py` (新增):
   - RoPE 实现对比测试，验证 Neuron 实现与原始 diffusers 实现等价

6. `test_rope_fix.py` (新增):
   - RoPE img_shapes 格式处理测试

7. `run_debug_component.py` (新增):
   - 组件隔离测试脚本，可单独测试 Neuron Transformer/VAE/Vision Encoder

8. `setup_nvme.sh`:
   - 修改为支持安全重新挂载，不会意外格式化已有数据

### 修复 5: RoPE (Rotary Position Embedding) 实现错误

**问题**：
Transformer 在实际推理中产生错误输出，尽管单独测试时 cosine similarity 接近 1.0。

详细调试发现 `apply_rotary_emb_neuron` 函数的实现有两个问题：

1. **img_shapes 格式解析错误**：
   - 编译时使用 `[(2, 32, 32)]` 格式
   - 运行时 pipeline 产生 `[[(1, 32, 32), (1, 32, 32)]]` 格式
   - 旧代码只提取第一个元组 `(1, 32, 32)`，导致只生成 1024 patches 的 RoPE，而不是 2048

2. **cos/sin 扩展方式错误**：
   - 旧代码使用 `cos.repeat(1, 1, 1, 2)` 将 `[c0, c1, ..., c31]` 扩展为 `[c0, c1, ..., c31, c0, c1, ..., c31]` (连接)
   - 正确方式应该是 `[c0, c0, c1, c1, ..., c31, c31]` (交错)

**证据**：
```
Testing apply_rotary_emb equivalence (修复前)
  Cosine similarity: 0.654633  ← 严重不匹配!
  Max absolute diff: 8.507820e+00

Testing apply_rotary_emb equivalence (修复后)
  Cosine similarity: 1.000152  ← 完美匹配!
  Max absolute diff: 4.768372e-07
```

**修复**：

**文件**: `neuron_qwen_image_edit/neuron_rope.py`

1. 修复 `NeuronQwenEmbedRope.forward()` 的 img_shapes 解析：
```python
# 正确处理多种格式:
# - (T, H, W): 单个元组
# - [(T, H, W)]: 列表中一个元组
# - [(T1, H, W), (T2, H, W)]: 多个元组，需要累加 frames
# - [[(T1, H, W), (T2, H, W)]]: 嵌套列表 (batch)

if isinstance(video_fhw, list) and len(video_fhw) > 0:
    first_elem = video_fhw[0]
    if isinstance(first_elem, tuple) and isinstance(first_elem[0], int):
        # 格式 2 或 3: 累加所有元组的 frames
        frame = sum(t[0] for t in video_fhw)
        height, width = first_elem[1], first_elem[2]
    elif isinstance(first_elem, (list, tuple)):
        # 格式 4: 嵌套列表，累加内层所有元组的 frames
        shapes = first_elem
        frame = sum(t[0] for t in shapes)
        height, width = shapes[0][1], shapes[0][2]
```

2. 修复 `apply_rotary_emb_neuron()` 的 cos/sin 扩展：
```python
# 使用 repeat_interleave 进行正确的交错扩展
# [c0, c1, ..., c31] -> [c0, c0, c1, c1, ..., c31, c31]
cos = cos.repeat_interleave(2, dim=-1)  # [S, D]
sin = sin.repeat_interleave(2, dim=-1)  # [S, D]

# 创建旋转版本 x_rotated = [-x_imag, x_real] (交错)
x_reshape = x.view(orig_shape[0], orig_shape[1], orig_shape[2], -1, 2)
x_rotated = torch.cat([-x_reshape[..., 1:2], x_reshape[..., 0:1]], dim=-1)
x_rotated = x_rotated.view(orig_shape)

# 应用旋转
out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
```

**重要**: 修复后需要**重新编译 Transformer**:
```bash
rm -rf /opt/dlami/nvme/compiled_models/transformer
python neuron_qwen_image_edit/compile_transformer.py \
    --height 512 --width 512 \
    --max_sequence_length 512 \
    --batch_size 1 --tp_degree 8 \
    --patch_multiplier 2 \
    --compiled_models_dir /opt/dlami/nvme/compiled_models
```

## 推荐配置

根据测试结果，推荐的运行配置：

| 组件 | 运行位置 | 原因 |
|------|----------|------|
| Transformer | Neuron (TP=8) | 计算量大，Neuron 加速显著 |
| VAE | Neuron | 计算量大，Neuron 加速显著 |
| Vision Encoder | CPU | 编译精度损失会被 LM 放大，CPU 更稳定 |
| Language Model | CPU | GQA 头数 (28Q/4KV) 与 TP=8 不对齐，CPU 避免问题 |

## 运行推理

### 推荐命令 (VAE + Transformer 在 Neuron, Text Encoder 在 CPU)

```bash
python run_qwen_image_edit.py \
    --images image1.png \
    --prompt "把女生变成男生" \
    --image_size 224 \
    --compiled_height 512 \
    --compiled_width 512 \
    --cpu_vision_encoder \
    --num_inference_steps 20
```

### 仅测试 Transformer 修复

```bash
python run_debug_component.py \
    --images image1.png \
    --prompt "把女生变成男生" \
    --use_neuron_transformer \
    --num_inference_steps 10
```

## 调试建议

如果仍然遇到问题：

1. **检查 pixel_values 形状**:
   ```python
   print(f"pixel_values: {model_inputs.pixel_values.shape}")
   # 应该是 (256, 1176) 对于 image_size=224
   ```

2. **检查 image_grid_thw**:
   ```python
   print(f"image_grid_thw: {model_inputs.image_grid_thw}")
   # 应该是 [[1, 16, 16]] 对于 image_size=224
   ```

3. **运行 RoPE 测试**:
   ```bash
   python test_rope_comparison.py
   # 应该显示 [PASS] Neuron RoPE matches Original RoPE
   ```

4. **运行多模态测试**:
   ```bash
   python tests/test_multimodal.py --image_size 224
   ```

5. **检查 position_ids**:
   在 `_get_rope_index()` 方法中添加打印语句查看计算的 position_ids
