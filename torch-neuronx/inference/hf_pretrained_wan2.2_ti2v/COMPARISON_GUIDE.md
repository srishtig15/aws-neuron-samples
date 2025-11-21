# GPU vs Trainium 对比指南

本指南说明如何对比GPU和Trainium上Wan2.2模型的推理结果。

## 概述

已实现固定随机种子功能，使得GPU和Trainium的推理结果可复现和可对比。

**固定参数：**
- 随机种子: `SEED = 42`
- 提示词: "A cat walks on the grass, realistic"
- 分辨率: 512x512
- 帧数: 81 (实际输出84帧)
- 推理步数: 50
- Guidance scale: 5.0

## 文件说明

- `run_wan2.2_ti2v_latency_optimized.py`: Trainium版本推理脚本（已添加固定种子）
- `run_wan2.2_ti2v_gpu.py`: GPU版本推理脚本（相同的种子和参数）
- `compare_outputs.py`: 数值对比工具

## 使用步骤

### 1. 在Trainium上运行推理

```bash
# 确保已编译模型
python neuron_wan2_2_ti2v/compile_text_encoder.py --compiled_models_dir compile_workdir_latency_optimized
python neuron_wan2_2_ti2v/compile_transformer_latency_optimized.py --compiled_models_dir compile_workdir_latency_optimized
python neuron_wan2_2_ti2v/compile_decoder.py --compiled_models_dir compile_workdir_latency_optimized --height 512 --width 512

# 运行推理
python run_wan2.2_ti2v_latency_optimized.py
```

输出：
- `output.mp4`: 生成的视频
- 控制台会显示推理时间

### 2. 在GPU上运行推理

```bash
# 在有GPU的机器上运行
python run_wan2.2_ti2v_gpu.py
```

输出：
- `output_gpu.mp4`: GPU生成的视频
- 控制台会显示推理时间和GPU信息

### 3. 视觉对比

直接播放两个视频文件：

```bash
# Linux
vlc output.mp4 output_gpu.mp4

# macOS
open output.mp4 output_gpu.mp4
```

观察：
- 内容是否一致（物体、动作、细节）
- 质量是否相似
- 是否有明显的artifacts或差异

### 4. 数值对比（可选）

如果需要精确的数值对比：

#### 4.1 修改脚本保存numpy数组

在两个推理脚本中，取消以下行的注释：

**在 `run_wan2.2_ti2v_latency_optimized.py` 中:**
```python
# 找到这几行并取消注释
npy.save("output_trainium.npy", output)
print("Numpy数组已保存到: output_trainium.npy")
```

**在 `run_wan2.2_ti2v_gpu.py` 中:**
```python
# 找到这几行并取消注释
npy.save("output_gpu.npy", output)
print("Numpy数组已保存到: output_gpu.npy")
```

#### 4.2 重新运行推理

```bash
# Trainium
python run_wan2.2_ti2v_latency_optimized.py

# GPU (在GPU机器上)
python run_wan2.2_ti2v_gpu.py
```

#### 4.3 运行对比工具

```bash
python compare_outputs.py output_gpu.npy output_trainium.npy
```

对比工具会输出：
- 统计信息（min, max, mean, std）
- 差异分析（绝对差异、相对差异）
- MSE和PSNR
- 相关系数
- 逐帧差异分析
- 结论和建议

## 预期结果

### 完全一致的情况（理想）
- 最大差异 < 0.01
- 相关系数 > 0.99
- PSNR > 40 dB

### 可接受的差异
- 最大差异 < 0.1
- 相关系数 > 0.95
- PSNR > 30 dB

差异来源可能包括：
1. **数值精度**: GPU使用FP32, Trainium可能使用BF16
2. **运算顺序**: 不同硬件可能有不同的运算实现
3. **编译器优化**: Neuron编译器的优化可能改变计算顺序
4. **硬件架构**: 不同硬件的浮点运算可能有细微差异

### 不可接受的差异
- 最大差异 >= 0.5
- 相关系数 < 0.9
- 视觉上有明显不同（内容、质量）

如果出现不可接受的差异，可能的问题：
1. 随机种子设置不正确
2. 模型编译有误
3. DecoderWrapper的feat_cache传播有问题
4. 数据类型转换有误

## 性能对比

推理时间对比：

| 平台 | Warmup时间 | 推理时间 | 硬件配置 |
|------|-----------|---------|---------|
| Trainium | ? | ? | trn1.2xlarge (16 NeuronCores) |
| GPU | ? | ? | (填写您的GPU型号) |

注意事项：
- Warmup时间包含模型加载和第一次推理
- 实际推理时间是第二次运行的时间
- Trainium使用编译后的模型（ahead-of-time compilation）
- GPU使用eager模式或JIT

## 调试提示

### 如果生成的视频帧数不对

检查：
```python
# 在两个脚本中确认
num_frames=81  # 输入帧数
# 期望输出: 84帧 (因为 VAE decoder 4x upsampling: 21 latent frames → 84 frames)
```

### 如果随机种子不生效

确认：
1. `set_seed(SEED)` 在主程序最开始被调用
2. `generator` 使用相同的种子
3. 没有其他代码在设置种子后使用随机数

### 如果GPU显存不足

可以尝试：
```python
# 修改 run_wan2.2_ti2v_gpu.py
DTYPE = torch.float16  # 从 bfloat16 改为 float16
# 或
pipe.enable_attention_slicing()  # 启用attention slicing
pipe.vae.enable_slicing()  # 启用VAE slicing
```

## 常见问题

**Q: 为什么warmup和主推理使用不同的种子？**

A: Warmup主要用于模型预热和时间测量，使用不同的种子(`SEED + 1000`)避免影响主推理的结果。主推理使用固定种子(`SEED`)以确保可复现性。

**Q: 可以使用其他的随机种子吗？**

A: 可以，只需在两个脚本的开头修改 `SEED = 42` 为其他值即可。确保两个脚本使用相同的种子。

**Q: 数值对比的阈值是如何确定的？**

A: 这些阈值基于经验：
- < 0.01: 非常接近（通常是完全一致）
- < 0.05: 较为接近（数值精度差异）
- < 0.1: 有一定差异（但在可接受范围内）
- >= 0.1: 明显差异（需要调查）

**Q: 为什么视频是84帧而不是81帧？**

A: 因为VAE decoder的时间上采样：
- 输入: 81帧 → encoder → 21 latent frames (4x下采样)
- latent: 21帧 → decoder → 84帧 (4x上采样)
- 84 = 21 × 4，这是正确的行为

## 进一步优化

如果需要更严格的一致性：

1. **使用相同的数据类型**:
   ```python
   # 两个脚本都使用 float32
   DTYPE = torch.float32
   ```

2. **启用确定性行为**:
   ```python
   # 在 set_seed() 函数中取消注释
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

3. **环境变量**:
   ```bash
   export PYTHONHASHSEED=42
   export CUBLAS_WORKSPACE_CONFIG=:4096:8  # GPU only
   ```

注意：这些设置可能会降低性能。

## 参考资料

- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
