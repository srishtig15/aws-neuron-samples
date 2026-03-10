#!/usr/bin/env python3
"""Patch diffusers to support flash-attn-4 (CuTeDSL) on CUDA 13 + Hopper GPUs.

flash-attn-4 (pip install --pre flash-attn-4) uses CuTeDSL for runtime JIT
compilation, eliminating the need for pre-built CUDA wheels. This makes it the
easiest way to get FlashAttention on CUDA 13 where flash-attn v2 has no wheels.

This script patches three diffusers files to add a "_flash_4" attention backend:
  1. utils/import_utils.py   - detect flash_attn.cute availability
  2. utils/__init__.py        - export the detection function
  3. models/attention_dispatch.py - register _flash_4 backend

Usage:
  python patch_diffusers_fa4.py                          # auto-detect diffusers path
  python patch_diffusers_fa4.py /path/to/diffusers       # explicit path
  python patch_diffusers_fa4.py --check                  # verify patch status

After patching, run inference with:
  export DIFFUSERS_ATTN_BACKEND="_flash_4"
  python run_wan2.2_t2v_a14b_gpu.py --resolution 480P
"""
import argparse
import importlib
import importlib.util
import os
import sys


def find_diffusers_path():
    """Auto-detect diffusers installation path."""
    spec = importlib.util.find_spec("diffusers")
    if spec is None or spec.origin is None:
        print("ERROR: diffusers is not installed in the current environment.")
        sys.exit(1)
    return os.path.dirname(spec.origin)


def check_status(diffusers_dir):
    """Check if the patch has already been applied."""
    import_utils = os.path.join(diffusers_dir, "utils", "import_utils.py")
    dispatch = os.path.join(diffusers_dir, "models", "attention_dispatch.py")
    init = os.path.join(diffusers_dir, "utils", "__init__.py")

    with open(import_utils) as f:
        iu = f.read()
    with open(dispatch) as f:
        dp = f.read()
    with open(init) as f:
        ini = f.read()

    patched_iu = "_flash_attn_4_available" in iu
    patched_dp = "_FLASH_4" in dp
    patched_init = "is_flash_attn_4_available" in ini

    if patched_iu and patched_dp and patched_init:
        print("Status: PATCHED (all 3 files)")
        return True
    elif not patched_iu and not patched_dp and not patched_init:
        print("Status: NOT PATCHED")
        return False
    else:
        print(f"Status: PARTIAL (import_utils={patched_iu}, dispatch={patched_dp}, init={patched_init})")
        return False


def patch_import_utils(diffusers_dir):
    """Add flash-attn-4 detection to import_utils.py."""
    path = os.path.join(diffusers_dir, "utils", "import_utils.py")
    with open(path) as f:
        content = f.read()

    if "_flash_attn_4_available" in content:
        print(f"  [SKIP] {path} (already patched)")
        return

    # Add detection after flash_attn_3 detection
    old = '_flash_attn_3_available, _flash_attn_3_version = _is_package_available("flash_attn_3")'
    new = old + """

# flash-attn-4 (CuTeDSL-based, supports CUDA 13 + Hopper/Blackwell)
_flash_attn_4_available = False
try:
    from flash_attn.cute import flash_attn_func as _fa4_test  # noqa: F401
    _flash_attn_4_available = True
    del _fa4_test
except (ImportError, ModuleNotFoundError):
    pass"""
    assert old in content, f"Cannot find flash_attn_3 detection in {path}"
    content = content.replace(old, new)

    # Add is_flash_attn_4_available() function
    old = """def is_flash_attn_3_available():
    return _flash_attn_3_available


def is_aiter_available():"""
    new = """def is_flash_attn_3_available():
    return _flash_attn_3_available


def is_flash_attn_4_available():
    return _flash_attn_4_available


def is_aiter_available():"""
    assert old in content, f"Cannot find is_flash_attn_3_available in {path}"
    content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK] {path}")


def patch_utils_init(diffusers_dir):
    """Export is_flash_attn_4_available from utils/__init__.py."""
    path = os.path.join(diffusers_dir, "utils", "__init__.py")
    with open(path) as f:
        content = f.read()

    if "is_flash_attn_4_available" in content:
        print(f"  [SKIP] {path} (already patched)")
        return

    old = "is_flash_attn_3_available,"
    new = "is_flash_attn_3_available,\n    is_flash_attn_4_available,"
    assert old in content, f"Cannot find is_flash_attn_3_available export in {path}"
    content = content.replace(old, new, 1)  # replace first occurrence only

    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK] {path}")


def patch_attention_dispatch(diffusers_dir):
    """Add _flash_4 backend to attention_dispatch.py."""
    path = os.path.join(diffusers_dir, "models", "attention_dispatch.py")
    with open(path) as f:
        content = f.read()

    if "_FLASH_4" in content:
        print(f"  [SKIP] {path} (already patched)")
        return

    # 1. Add import
    old = "    is_flash_attn_3_available,\n    is_flash_attn_available,"
    new = "    is_flash_attn_3_available,\n    is_flash_attn_4_available,\n    is_flash_attn_available,"
    assert old in content, f"Cannot find flash_attn imports in {path}"
    content = content.replace(old, new)

    # 2. Add availability check and import
    old = """if _CAN_USE_AITER_ATTN:
    try:
        from aiter import flash_attn_func as aiter_flash_attn_func"""
    new = """_CAN_USE_FLASH_ATTN_4 = is_flash_attn_4_available()

if _CAN_USE_FLASH_ATTN_4:
    try:
        from flash_attn.cute import flash_attn_func as flash_attn_4_func
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning(f"flash_attn_4 (CuTeDSL) failed to import: {e}. Falling back to native attention.")
        _CAN_USE_FLASH_ATTN_4 = False
        flash_attn_4_func = None
else:
    flash_attn_4_func = None


if _CAN_USE_AITER_ATTN:
    try:
        from aiter import flash_attn_func as aiter_flash_attn_func"""
    assert old in content, f"Cannot find _CAN_USE_AITER_ATTN block in {path}"
    content = content.replace(old, new)

    # 3. Add enum value
    old = """    # `aiter`
    AITER = "aiter\""""
    new = """    _FLASH_4 = "_flash_4"

    # `aiter`
    AITER = "aiter\""""
    assert old in content, f"Cannot find AITER enum in {path}"
    content = content.replace(old, new)

    # 4. Add requirement check
    old = """    elif backend in [
        AttentionBackendName.FLASH_HUB,
        AttentionBackendName.FLASH_VARLEN_HUB,"""
    new = """    elif backend == AttentionBackendName._FLASH_4:
        if not _CAN_USE_FLASH_ATTN_4:
            raise RuntimeError(
                f"Flash Attention 4 backend '{backend.value}' is not usable. "
                "Please install with: pip install --pre flash-attn-4"
            )

    elif backend in [
        AttentionBackendName.FLASH_HUB,
        AttentionBackendName.FLASH_VARLEN_HUB,"""
    assert old in content, f"Cannot find FLASH_HUB requirement check in {path}"
    content = content.replace(old, new)

    # 5. Register backend (after _flash_attention_3)
    old = """@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_3_HUB,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _flash_attention_3_hub("""
    new = """@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_4,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _flash_attention_4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    return_lse: bool = False,
    _parallel_config: "ParallelConfig" | None = None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for flash-attn 4.")
    if _parallel_config is not None:
        raise ValueError("Context parallel is not yet supported for flash-attn 4.")

    # flash-attn-4 API: (q, k, v, softmax_scale, causal, ..., return_lse)
    # Note: no dropout_p parameter in flash-attn-4
    # flash-attn-4 may return (out, lse) tuple regardless of return_lse flag
    result = flash_attn_4_func(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
        return_lse=True,
    )
    if isinstance(result, tuple):
        out, lse = result[0], result[1]
    else:
        out, lse = result, None
    return (out, lse) if return_lse else out


@_AttentionBackendRegistry.register(
    AttentionBackendName._FLASH_3_HUB,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _flash_attention_3_hub("""
    assert old in content, f"Cannot find _flash_attention_3_hub in {path}"
    content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK] {path}")


def main():
    parser = argparse.ArgumentParser(description="Patch diffusers for flash-attn-4 support")
    parser.add_argument("diffusers_dir", nargs="?", default=None,
                        help="Path to diffusers package directory (auto-detected if omitted)")
    parser.add_argument("--check", action="store_true", help="Check patch status only")
    args = parser.parse_args()

    diffusers_dir = args.diffusers_dir or find_diffusers_path()
    print(f"Diffusers path: {diffusers_dir}")

    if args.check:
        check_status(diffusers_dir)
        return

    print("Patching diffusers for flash-attn-4 support...")
    patch_import_utils(diffusers_dir)
    patch_utils_init(diffusers_dir)
    patch_attention_dispatch(diffusers_dir)

    print("\nDone! To use flash-attn-4:")
    print('  export DIFFUSERS_ATTN_BACKEND="_flash_4"')
    print("  python run_wan2.2_t2v_a14b_gpu.py --resolution 480P")


if __name__ == "__main__":
    main()
