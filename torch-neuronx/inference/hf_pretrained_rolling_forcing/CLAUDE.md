# RollingForcing Trn2 项目状态 (2026-02-28)

## 项目位置
- Trn2: ~/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_rolling_forcing/
- GPU参考: H100 (Host alias: H100), /tmp/RollingForcing/
- SSH alias: Trn2 (ubuntu), H100 (ubuntu)
- Trn2 venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
- H100 venv: source /opt/pytorch/bin/activate
- Wan2.2 参考代码: /Users/henanwan/Documents/workspace/wan2.2-ti2v-neuron/code/

## 编译状态 - 全部完成
- Text encoder: TP=8, parallel_model_trace API
- VAE decoder NoCache: TP=8, WanUpsample patched (nearest-exact -> nearest)
- post_quant_conv: world_size=8
- Transformer: TP=4, world_size=8, 当前使用NON-CAUSAL attention (已恢复备份)
- 编译产物在 Trn2 compiled_models/

## 推理状态 - Pipeline跑通，输出质量待改进

### 性能
- Denoising: ~22s (11 windows × 2.0s/window)
- Total: ~43s (text encode ~10.7s + denoise ~22s + VAE decode ~9.9s)

### 质量分析 (2026-02-28 最新)

GPU对比实验在H100上完成:
| 方法 | pixel_std | latent cos_sim vs original |
|------|-----------|---------------------------|
| 原始GPU (KV cache) | 70.2 | 1.0 (reference) |
| Neuron padded, CFG=7 | 56.8 | - |
| GPU bidirectional, no pad | 22.4 | 0.186 |
| GPU block-causal, no pad | 22.8 | 0.177 |
| GPU token-causal, no pad | ~14 | 0.139 |

**关键结论**:
1. Padding dilution不是主要问题 (去掉padding反而更差)
2. 注意力mask类型不影响质量 (bidirectional/causal/block-causal结果一致~22)
3. KV cache是质量的关键 - 提供clean frame的隔离表示
4. 需要实现真正的KV cache推理才能匹配原始质量

### 根本原因
原始pipeline每个window做2次model call:
1. Denoise: 读取cached clean K/V，生成denoised frames
2. Cache update: 将clean frames重新过模型(t≈0)，计算isolated K/V存入cache

我们的full-sequence方案只做1次call:
- Clean+noisy frames一起处理
- 即使用block-causal mask，clean frames的K/V也是重新计算的(非cached isolated版本)
- 模型训练时期望的是KV cache inference pattern

### 下一步: KV Cache实现方案

NKI flash_fwd kernel分析:
- 支持Q≠KV长度 (用于cross-attention已验证)
- logit_bias: [1,1,seq_q,seq_k] (batch/head broadcast), 但需要完整Q×K矩阵，对32760太大
- use_causal_mask=True支持，但不等同于block-causal
- sliding_window支持 (causal sliding)

可选方案:
A) 外部KV cache - 每层K/V作为model I/O (30层×2 tensors)
B) 自定义NKI kernel - 添加kv_valid_len参数
C) Bucketed编译 - 不同KV长度各编译一个model
D) 修改NKI kernel支持block-causal mask

## 关键技术细节
- CausalWanModel constructor: model_type, patch_size, text_len, in_dim, dim, ffn_dim等
- FSDP checkpoint: 需要strip _fsdp_wrapped_module. 和 model. 前缀
- flex_attention在causal_model.py中被注释掉(line 11)
- _forward_train用flex_attention+block_mask, _forward_inference用KV cache+flash_attn
- 原始用flash_attn causal=False (因果性通过cache结构保证)
- Anchor re-roping: block 0在cache中存un-roped keys, 每次使用时动态re-rope
- kv_cache_size = 1560 * 24 = 37440 (24帧), 7 blocks × 4680 = 32760不需eviction
- NKI attention_isa_kernel: 非causal, 支持不同Q/KV长度
- NKI flash_fwd: 支持causal/sliding_window/logit_bias, tile_size=2048
