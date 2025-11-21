"""
对比GPU和Trainium生成的视频输出

使用方法：
    1. 先分别运行 GPU 和 Trainium 版本的推理脚本，保存numpy数组
    2. 运行此脚本进行数值对比

示例：
    python compare_outputs.py output_gpu.npy output_trainium.npy
"""

import numpy as np
import sys
import torch


def compare_outputs(gpu_path: str, trainium_path: str):
    """
    对比GPU和Trainium的输出

    Args:
        gpu_path: GPU输出的numpy文件路径
        trainium_path: Trainium输出的numpy文件路径
    """
    print("="*80)
    print("GPU vs Trainium 输出对比")
    print("="*80)

    # 加载数据
    print(f"\n加载数据...")
    gpu_output = np.load(gpu_path)
    trainium_output = np.load(trainium_path)

    print(f"GPU输出 shape: {gpu_output.shape}, dtype: {gpu_output.dtype}")
    print(f"Trainium输出 shape: {trainium_output.shape}, dtype: {trainium_output.dtype}")

    # 检查shape是否一致
    if gpu_output.shape != trainium_output.shape:
        print(f"\n⚠️  警告: Shape不一致！")
        print(f"   GPU: {gpu_output.shape}")
        print(f"   Trainium: {trainium_output.shape}")
        return

    print(f"\n✓ Shape一致: {gpu_output.shape}")

    # 计算统计信息
    print("\n" + "="*80)
    print("统计信息对比")
    print("="*80)

    print("\nGPU输出:")
    print(f"  Min: {gpu_output.min():.6f}")
    print(f"  Max: {gpu_output.max():.6f}")
    print(f"  Mean: {gpu_output.mean():.6f}")
    print(f"  Std: {gpu_output.std():.6f}")

    print("\nTrainium输出:")
    print(f"  Min: {trainium_output.min():.6f}")
    print(f"  Max: {trainium_output.max():.6f}")
    print(f"  Mean: {trainium_output.mean():.6f}")
    print(f"  Std: {trainium_output.std():.6f}")

    # 计算差异
    print("\n" + "="*80)
    print("差异分析")
    print("="*80)

    diff = np.abs(gpu_output - trainium_output)

    print(f"\n绝对差异 (Absolute Difference):")
    print(f"  Min: {diff.min():.6f}")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std: {diff.std():.6f}")

    # 相对差异（避免除以0）
    epsilon = 1e-8
    rel_diff = diff / (np.abs(gpu_output) + epsilon)

    print(f"\n相对差异 (Relative Difference):")
    print(f"  Mean: {rel_diff.mean():.6f}")
    print(f"  Median: {np.median(rel_diff):.6f}")
    print(f"  95th percentile: {np.percentile(rel_diff, 95):.6f}")
    print(f"  99th percentile: {np.percentile(rel_diff, 99):.6f}")

    # MSE, PSNR
    mse = np.mean((gpu_output - trainium_output) ** 2)
    print(f"\nMSE (Mean Squared Error): {mse:.8f}")

    # PSNR (假设值范围是0-1)
    if mse > 0:
        max_val = max(gpu_output.max(), trainium_output.max())
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        print(f"PSNR: {psnr:.2f} dB")

    # 相关系数
    correlation = np.corrcoef(gpu_output.flatten(), trainium_output.flatten())[0, 1]
    print(f"相关系数: {correlation:.6f}")

    # 逐帧差异分析（如果是视频）
    if len(gpu_output.shape) == 4:  # (frames, height, width, channels)
        print("\n" + "="*80)
        print("逐帧差异分析")
        print("="*80)

        num_frames = gpu_output.shape[0]
        frame_diffs = []

        for i in range(num_frames):
            frame_diff = np.abs(gpu_output[i] - trainium_output[i]).mean()
            frame_diffs.append(frame_diff)

        frame_diffs = np.array(frame_diffs)

        print(f"\n帧数: {num_frames}")
        print(f"平均帧差异: {frame_diffs.mean():.6f}")
        print(f"最大帧差异: {frame_diffs.max():.6f} (帧 {frame_diffs.argmax()})")
        print(f"最小帧差异: {frame_diffs.min():.6f} (帧 {frame_diffs.argmin()})")

        # 找出差异最大的几帧
        top_k = 5
        top_diff_frames = np.argsort(frame_diffs)[-top_k:][::-1]
        print(f"\n差异最大的{top_k}帧:")
        for idx in top_diff_frames:
            print(f"  帧 {idx}: 平均差异 {frame_diffs[idx]:.6f}")

    # 结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if diff.max() < 0.01:
        print("\n✓ 输出非常接近（最大差异 < 0.01）")
    elif diff.max() < 0.05:
        print("\n✓ 输出较为接近（最大差异 < 0.05）")
    elif diff.max() < 0.1:
        print("\n⚠️  输出有一定差异（最大差异 < 0.1）")
    else:
        print("\n⚠️  输出有明显差异（最大差异 >= 0.1）")

    if correlation > 0.99:
        print("✓ 相关性很高（> 0.99）")
    elif correlation > 0.95:
        print("✓ 相关性较高（> 0.95）")
    else:
        print("⚠️  相关性较低（< 0.95）")

    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python compare_outputs.py <gpu_output.npy> <trainium_output.npy>")
        print("\n提示: 需要先修改推理脚本，在保存视频前添加:")
        print("  np.save('output_gpu.npy', output)")
        print("  或")
        print("  np.save('output_trainium.npy', output)")
        sys.exit(1)

    gpu_path = sys.argv[1]
    trainium_path = sys.argv[2]

    compare_outputs(gpu_path, trainium_path)
