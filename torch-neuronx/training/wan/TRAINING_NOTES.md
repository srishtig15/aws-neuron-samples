# Wan2.1 Neuron Training 配置和问题解决记录

## 日期
2025-10-10

## 硬件环境
- 实例类型: **trn2.48xlarge**
- NeuronCore 数量: **64** (16 devices × 4 cores, LNC=2)
- 每个设备 HBM: **96 GB**
- Neuron 编译器版本: **2.21.18209.0+043b1bf7**

## 训练配置总结

### 最终可用配置
```bash
# 当前可以成功编译和训练的配置
python run.py \
  --tensor_parallel_degree 8 \
  --max_frames 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --train_height 128 \
  --train_width 128 \
  --epochs 6
```

### 关键参数说明
| 参数 | 值 | 说明 |
|------|-----|------|
| `nproc_per_node` | 64 | 使用所有 NeuronCore |
| `tensor_parallel_degree` | 8 | 模型分片到 8 个核心 (TP=8) |
| `max_frames` | 2 | 视频帧数（内存限制） |
| `train_height/width` | 128 | 训练分辨率（可逐步提升） |
| `batch_size` | 1 | 每设备批次大小 |
| `gradient_accumulation_steps` | 2 | 梯度累积步数 |

### 有效数据并行度
```
data_parallel_degree = nproc_per_node / tensor_parallel_degree
                     = 64 / 8 = 8
```

## 遇到的问题和解决方案

### 问题 1: 编译器参数不支持
**错误**: `Unknown command line argument '--enable-hbm-spilling'`

**原因**: Neuron 编译器版本 2.21 不支持此参数

**解决**: 移除该参数，依赖其他内存优化策略

### 问题 2: SB (Scratchpad Buffer) 内存分配失败
**错误**: `[GCA030] Couldn't allocate every tensor in SB and spilling can't help`

**原因**:
- 训练需要保存所有中间激活值用于反向传播
- 内存需求是推理的 2-3 倍
- TP=4 时每个核心处理 1/4 模型，SB 压力过大

**解决方案**:
1. ✅ **增加 TP degree**: 4 → 8 (每个核心处理更小的模型)
2. ✅ **降低分辨率**: 256×256 → 128×128
3. ✅ **减少帧数**: 4 → 2
4. ✅ **使用优化标志**: `--internal-hlo2tensorizer-options='--fuse-dot-logistic=false'`

### 问题 3: 模型保存失败

#### 3.1 XLA 存储指针错误
**错误**: `RuntimeError: Attempted to access the data pointer on an invalid python storage`

**原因**: XLA 设备上的张量无法直接被 safetensors 序列化

**解决**: 修改 `save_pipeline()` 函数
```python
# 关键步骤:
1. 获取 state_dict
2. 使用 xm._maybe_convert_to_cpu() 转换到 CPU
3. 使用 torch.save() 保存（不用 safetensors）
4. 分别保存各个组件
```

#### 3.2 FrozenDict 配置保存错误
**错误**: `AttributeError: 'FrozenDict' object has no attribute 'save_pretrained'`

**原因**: `pipe.transformer.config` 是 `FrozenDict` 对象，不是 `PretrainedConfig`

**解决**: 手动转换为字典并保存为 JSON
```python
# 错误方式:
pipe.transformer.config.save_pretrained(transformer_save_path)

# 正确方式:
config_dict = dict(pipe.transformer.config)
with open(os.path.join(transformer_save_path, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)
```

#### 3.3 VAE 和 Text Encoder 在 XLA 设备上
**问题**: 这些组件可能也在 XLA 设备上，需要先移到 CPU

**解决**:
```python
# 移到 CPU 再保存
vae_cpu = pipe.vae.to('cpu')
vae_cpu.save_pretrained(os.path.join(results_dir, "vae"))
del vae_cpu  # 释放内存

# 对所有组件都添加异常处理
try:
    text_encoder_cpu = pipe.text_encoder.to('cpu')
    text_encoder_cpu.save_pretrained(os.path.join(results_dir, "text_encoder"))
    del text_encoder_cpu
except Exception as e:
    xm.master_print(f"Warning: Could not save text_encoder: {e}")
```

## 内存优化策略总结

### SB (Scratchpad Buffer) 是什么？
- **片上高速内存**，容量小（几百 MB）但速度快
- 类比：快餐店的厨房工作台
- **HBM**：后厨冷藏室（96 GB，较慢）
- **Host RAM**：仓库（最大但最慢）

### 内存占用估算
```
总 SB 需求 = 模型大小 × 数据大小 × 训练开销

当前配置:
- 模型: 1/8 (TP=8)
- 数据: 2 frames, 128×128
- 训练: 2-3x overhead
结果: 可以编译 ✅

之前配置:
- 模型: 1/4 (TP=4)
- 数据: 4 frames, 256×256
结果: SB 溢出 ❌
```

## 高分辨率训练策略

### 方案 1: 渐进式分辨率训练 ⭐ 推荐

**阶段 1 - 低分辨率基础训练** (1-2天)
```bash
python run.py \
  --train_height 128 \
  --train_width 128 \
  --max_frames 2 \
  --tensor_parallel_degree 8 \
  --epochs 20
```

**阶段 2 - 中分辨率微调** (2-3天)
```bash
python run.py \
  --train_height 256 \
  --train_width 256 \
  --max_frames 2 \
  --tensor_parallel_degree 8 \
  --epochs 10 \
  --resume_from_checkpoint \
  --resume_checkpoint_step <STEP_FROM_STAGE1>
```

**阶段 3 - 高分辨率精细化** (3-5天)
```bash
python run.py \
  --train_height 512 \
  --train_width 512 \
  --max_frames 4 \
  --tensor_parallel_degree 16 \
  --epochs 5 \
  --resume_from_checkpoint \
  --resume_checkpoint_step <STEP_FROM_STAGE2>
```

### 方案 2: 增加 Tensor Parallelism
```bash
# 对于 512×512 高分辨率
python run.py \
  --tensor_parallel_degree 16 \
  --train_height 512 \
  --train_width 512 \
  --max_frames 2
```

**权衡**:
- ✅ 降低每个核心的内存压力
- ❌ 数据并行度降低 (64/16=4)
- ❌ 可能降低训练吞吐量

## 编译器标志说明

### 当前使用的编译器参数
```python
compiler_flags = """
  --target=trn2
  --lnc=2
  --retry_failed_compilation
  --cache_dir="./compiler_cache"
  --model-type=transformer
  --enable-saturate-infinity
  --internal-hlo2tensorizer-options='--fuse-dot-logistic=false'
"""
```

### 参数解释
- `--target=trn2`: 目标硬件 Trainium2
- `--lnc=2`: Logical NeuronCore Config (每个逻辑核=2个物理核)
- `--model-type=transformer`: 启用 Transformer 特定优化
- `--enable-saturate-infinity`: 数值稳定性（必需）
- `--fuse-dot-logistic=false`: 禁用矩阵乘法和激活函数融合（提高稳定性）

## 工具和命令

### 编译相关
```bash
# neuron_parallel_compile: 高层工具，自动提取图并并行编译
neuron_parallel_compile --num_parallel 10 <training_command>

# neuronx-cc compile: 底层编译器，手动编译单个 HLO 文件
neuronx-cc compile --framework XLA model.hlo --output model.neff
```

### 监控和调试
```bash
# 查看 Neuron 设备状态
neuron-ls

# 实时监控资源使用
neuron-top

# 查看编译缓存
ls -la compiler_cache/

# 查看训练日志
tail -f run_output.log
```

## 关键概念理解

### Tensor Parallelism (TP)
- 将**单个模型**切分到多个设备
- TP=8: 每个核心处理 1/8 的模型
- 适合**大模型**

### Data Parallelism (DP)
- 在多个设备上运行**模型副本**
- 每个副本处理不同数据
- DP = world_size / TP = 64 / 8 = 8

### Gradient Accumulation
- 在多个 micro-batch 上累积梯度
- 等效于更大的 batch size
- 节省内存，不需要同时处理大 batch

### 有效全局批次大小
```
Global Batch Size = batch_size × DP × gradient_accumulation_steps
                  = 1 × 8 × 2 = 16
```

## 训练流程

### 完整训练命令 (run.py)
```bash
python run.py \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --resolution 512 \
  --batch_size 1 \
  --max_frames 2 \
  --tensor_parallel_degree 8 \
  --gradient_accumulation_steps 2 \
  --train_height 128 \
  --train_width 128 \
  --epochs 6 \
  --checkpointing_steps 750 \
  --save_model_epochs 1
```

### 直接训练命令 (wan_neuron_origin.py)
```bash
torchrun --nproc_per_node=64 wan_neuron_origin.py \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --resolution 512 \
  --batch_size 1 \
  --max_frames 2 \
  --tensor_parallel_degree 8 \
  --gradient_accumulation_steps 2 \
  --train_height 128 \
  --train_width 128 \
  --epochs 6
```

## 检查点和模型保存

### 检查点位置
```
wan_<model>_training-<resolution>-batch<size>-AdamW-64w-zero1_optimizer-grad_checkpointing_results/
├── checkpoint-unet-epoch_X-step_Y-cumulative_train_step_Z.pt
└── checkpoint-optimizer-epoch_X-step_Y-cumulative_train_step_Z-rank_*.pt
```

### 模型保存位置
```
<results_dir>-EPOCH_N/
├── transformer/
│   ├── diffusion_pytorch_model.bin  # 模型权重
│   └── config.json                  # 模型配置
├── vae/
│   ├── diffusion_pytorch_model.safetensors
│   └── config.json
├── text_encoder/
│   ├── model.safetensors
│   └── config.json
├── tokenizer/
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── scheduler/
│   └── scheduler_config.json
└── model_index.json                 # Pipeline 元数据
```

### 完整的保存流程

修复后的 `save_pipeline()` 函数执行以下步骤：

1. **同步 XLA 操作**
```python
xm.mark_step()
xm.wait_device_ops()
```

2. **保存 Transformer**
```python
# 提取状态字典
transformer_state = pipe.transformer.state_dict()

# 转换到 CPU
cpu_transformer_state = {}
for k, v in transformer_state.items():
    cpu_transformer_state[k] = xm._maybe_convert_to_cpu(v)

# 保存权重
torch.save(cpu_transformer_state, "transformer/diffusion_pytorch_model.bin")

# 保存配置（手动转换 FrozenDict）
config_dict = dict(pipe.transformer.config)
with open("transformer/config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
```

3. **保存其他组件**（带异常处理）
```python
# 每个组件都先移到 CPU
vae_cpu = pipe.vae.to('cpu')
vae_cpu.save_pretrained("vae/")
del vae_cpu  # 释放内存

# 类似处理 text_encoder, tokenizer, scheduler
```

4. **保存 model_index.json**
```python
model_index = {
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.21.0",
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5Tokenizer"],
    "scheduler": ["diffusers", "DDPMScheduler"]
}
```

### 加载训练好的模型
```python
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(
    "/path/to/saved/model",
    torch_dtype=torch.bfloat16
)

# 生成视频
output = pipe(
    prompt="A cat playing with a ball",
    num_frames=16,
    height=512,
    width=512
)
```

### 验证保存的模型
```bash
# 检查文件结构
tree <results_dir>-EPOCH_N/

# 验证文件大小
du -sh <results_dir>-EPOCH_N/*

# 测试加载
python -c "
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained('<results_dir>-EPOCH_N/')
print('Model loaded successfully!')
"
```

## 性能优化建议

### 已启用的优化
- ✅ Gradient Checkpointing (line 403)
- ✅ ZeRO-1 Optimizer Sharding
- ✅ BFloat16 混合精度
- ✅ Stochastic Rounding
- ✅ XLA 图优化

### 未来可尝试
- [ ] 序列并行（Sequence Parallelism）
- [ ] 更激进的 Activation Checkpointing
- [ ] Flash Attention (如果支持)
- [ ] 优化数据加载管道

## 常见问题排查

### 编译失败
1. 检查 `compiler_cache/` 是否有足够空间
2. 查看 `log-neuron-cc.txt` 详细错误
3. 尝试增加 TP degree
4. 降低分辨率或帧数

### OOM (内存不足)
1. 降低 batch_size
2. 增加 TP degree
3. 减少 max_frames
4. 降低训练分辨率

### 训练不收敛
1. 检查学习率是否合适
2. 验证数据预处理
3. 查看 LOSSES-RANK-*.txt 文件
4. 可能需要从更低分辨率开始

## 参考资源

### AWS Neuron 文档
- [Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Training on Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/training-guide.html)

### 相关 GitHub Issue
- [aws-neuron-sdk issues](https://github.com/aws-neuron/aws-neuron-sdk/issues)

## 总结

通过以下关键调整，成功解决了编译和训练问题：

1. **TP degree**: 4 → 8 （降低每个核心的模型大小）
2. **分辨率**: 256×256 → 128×128 （降低数据大小）
3. **帧数**: 4 → 2 （进一步降低内存需求）
4. **编译器优化**: 使用 `--fuse-dot-logistic=false` 提高稳定性
5. **保存机制**: 修复 XLA 张量保存问题

**下一步**: 使用渐进式训练策略逐步提升分辨率到目标 512×512。

## 更新日志

### 2025-10-10 (第二次更新)
- ✅ 修复模型保存的 FrozenDict 错误
- ✅ 添加 VAE 和 Text Encoder 的 CPU 转换
- ✅ 为所有组件添加异常处理
- ✅ 完善保存流程文档
- ✅ 添加模型验证命令

### 2025-10-10 (初始版本)
- ✅ 解决 SB 内存分配问题
- ✅ 优化编译器参数
- ✅ 配置 Tensor Parallelism
- ✅ 创建渐进式训练策略

---

*最后更新: 2025-10-10*
*文档版本: v1.1*
