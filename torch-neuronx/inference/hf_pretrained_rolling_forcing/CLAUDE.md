# RollingForcing Trn2 项目状态 (2026-02-26)

## 项目位置
- Trn2: `~/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_rolling_forcing/`
- 本地: `/Users/henanwan/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_rolling_forcing/`
- SSH alias: `Trn2` (ubuntu), GPU: `P5EN-1` (ubuntu, 8xH100, venv `/opt/pytorch`)
- Trn2 venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Wan2.2 参考代码: `/Users/henanwan/Documents/workspace/wan2.2-ti2v-neuron/code/`

## 编译状态 - 全部完成
- Text encoder: TP=8, `parallel_model_trace` API
- VAE decoder NoCache: TP=8, world_size=8, `ModelBuilder` API, decoder_frames=2
- post_quant_conv: world_size=8
- Transformer: TP=4, world_size=8 (CP=2), `ModelBuilder` API
- 编译产物在 Trn2 `compiled_models/` 目录

## 推理状态 - Pipeline跑通但输出有问题
- 性能: denoising 22.2s (11 windows x 2.02s/window), total ~43s
- 问题: 生成视频是乱码

## 明天的TODO

### 1. 修复视频输出质量 (最重要)
- **VAE decoder temporal upsampling 缺失**: 当前 NoCache 模式按2帧decode，输出只有21帧而非81帧。feat_cache buffer在Neuron上可能没有正确跨call传播
- 可能方案:
  - A) 编译一个能一次性decode 21帧latent的decoder (输出81帧)
  - B) 修复buffer persistence，确保feat_cache跨call更新
  - C) 先在CPU上跑一次VAE decode对比输出shape验证问题

### 2. 验证 rolling forcing denoising 逻辑
- 对比GPU参考实现 (`/tmp/RollingForcing/pipeline/rolling_forcing_inference.py`) 的noisy_cache更新逻辑
- 检查timestep warping是否正确 (warped steps: 999.8, 982.5, 960.3, 931.1, 890.6 — 看起来偏高，可能应该递减到接近0)
- 在GPU上跑参考实现验证denoising输出

### 3. GPU Benchmark
- P5EN-1需要先下载Wan2.1基础模型到 `/tmp/RollingForcing/wan_models/Wan2.1-T2V-1.3B/`
- flash_attn与CUDA 13不兼容，已确认SDPA fallback可用 (uninstall flash_attn即可)
- RollingForcing checkpoint已下载到P5EN-1: `/tmp/rolling_forcing_ckpt/checkpoints/rolling_forcing_dmd.pt`
- benchmark脚本: `run_rolling_forcing_gpu.py`

### 4. 性能优化 (后续)
- Text encoding 10.7s偏慢，可能需要warmup
- VAE decoding 9.9s，优化多帧decode
- 探索warmup run稳定Neuron延迟

## 关键技术细节
- NxDModel加载: `NxDModel.load()` -> `set_weights(weights_list)` -> `to_neuron()`
- Transformer权重: 4个TP文件 -> `prepare_cp_checkpoints` 复制到8个rank
- Decoder权重: tp0复制8份 (`load_duplicated_weights`)
- Text encoder用不同API: `parallel_model_load` (不需要set_weights/to_neuron)
- 所有NxDModel必须同一world_size=8，否则NRT communicator冲突
- NKI flash attention要求seq len是2的幂次，已在attention中pad到32768
- Wan2.1 VAE: base_dim=96, 32 feat_cache entries (不是Wan2.2的34个)
- Decoder CACHE_T=2要求: latent temporal dim >= 2才能XLA trace
