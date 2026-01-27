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
   - 添加 processor 配置代码 (第 822-829 行)

2. `neuron_qwen_image_edit/neuron_commons.py`:
   - 添加 `image_token_id` 和 `vision_start_token_id` 属性 (第 107-109 行)
   - 添加 `_get_rope_index()` 方法 (第 111-192 行) - 逐 token 处理版本
   - 更新 `forward()` 方法使用新的 position_ids 计算

3. `tests/test_multimodal.py` (新增):
   - 多模态测试，验证图像+文本处理

4. `setup_nvme.sh`:
   - 修改为支持安全重新挂载，不会意外格式化已有数据

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

3. **运行多模态测试**:
   ```bash
   python tests/test_multimodal.py --image_size 224
   ```

4. **检查 position_ids**:
   在 `_get_rope_index()` 方法中添加打印语句查看计算的 position_ids
