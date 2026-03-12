"""
Wan2.2 T2V-A14B Inference with MoE (Mixture of Experts) Transformer Switching.

This model uses TWO transformers (both WanTransformer3DModel with 40 heads, 40 layers)
that are selected based on the denoising timestep:
- transformer (high-noise expert): used when timestep >= 875 (~12.5% of steps)
- transformer_2 (low-noise expert): used when timestep < 875 (~87.5% of steps)

Architecture:
- TP=4, CP=N for each transformer (CP auto-detected from compiled model config)
  - 480P: CP=2 (world_size=8)
  - 720P: CP=4 (world_size=16)
- Both transformers share the SAME compiled NEFF (identical architecture)
- MoE weight swap uses NxDModel.replace_weights() (no unload/reload needed)
- Staged pipeline:
  Phase 1: Text Encoder -> encode prompt -> unload
  Phase 2: Load 1 NxDModel -> high-noise steps -> replace_weights -> low-noise steps
  Phase 3: VAE Decode -> Neuron (chunked) or CPU fallback -> save video

Usage:
    # 480P (default)
    python run_wan2.2_t2v_a14b.py \\
        --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \\
        --prompt "A cat walking on the grass"

    # 720P
    python run_wan2.2_t2v_a14b.py \\
        --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b_720p \\
        --height 720 --width 1280 \\
        --prompt "A cat walking on the grass"
"""
# IMPORTANT: Set environment variables BEFORE any imports.
# Auto-detect world_size from compiled transformer config.
import json
import os
import sys

def _detect_world_size():
    """Read world_size from compiled transformer config before Neuron init."""
    compiled_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == "--compiled_models_dir" and i + 1 < len(sys.argv):
            compiled_dir = sys.argv[i + 1]
            break
    if compiled_dir is None:
        compiled_dir = "/opt/dlami/nvme/compiled_models_t2v_a14b"
    config_path = os.path.join(compiled_dir, "transformer", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)["world_size"]
    return 8

_world_size = _detect_world_size()
# Neuron environment setup. All env vars respect external values if already set.
# Usage: NEURON_RT_VISIBLE_CORES=8-15 NEURON_RT_NUM_CORES=8 python run_wan2.2_t2v_a14b.py ...
#
# On trn2.48xlarge with LNC=2: 16 chips, 4 logical NCs per chip, 24GB HBM per NC.
#   - 480P: world_size=8 -> 8 NCs = 2 chips; 720P: world_size=16 -> 16 NCs = 4 chips
_num_cores = int(os.environ.get("NEURON_RT_NUM_CORES", str(_world_size)))
os.environ.setdefault("NEURON_RT_NUM_CORES", str(_num_cores))
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
if "NEURON_RT_VISIBLE_CORES" not in os.environ:
    _core_start = 0
    if _num_cores > 8:
        _core_start = 64 - _num_cores  # Auto-offset for 720P to avoid conflicts
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{_core_start}-{_core_start + _num_cores - 1}"
os.environ.setdefault("NEURON_RT_INSPECT_ENABLE", "0")
os.environ.setdefault("NEURON_RT_INSPECT_DEVICE_PROFILE", "0")
os.environ.setdefault("NEURON_RT_INSPECT_SYSTEM_PROFILE", "0")
os.environ.setdefault("NEURON_RT_PROFILING_MODE", "0")
print(f"Neuron config: world_size={_world_size}, NUM_CORES={os.environ['NEURON_RT_NUM_CORES']}, "
      f"VCS={os.environ['NEURON_RT_VIRTUAL_CORE_SIZE']}, VISIBLE_CORES={os.environ['NEURON_RT_VISIBLE_CORES']}")

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import argparse
import gc
import numpy as np
import random
import time
import torch

# Subprocess mode: all Neuron phases run in subprocesses for clean HBM lifecycle.
# Main process never imports torch_neuronx — each subprocess initializes its own NRT.
_subprocess_mode = True

from safetensors.torch import load_file

from neuron_wan2_2_t2v_a14b.neuron_commons import (
    InferenceTextEncoderWrapperV2,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_model_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        if rank == 0:
            removed = len(raw_ckpt) - len(ckpt)
            if removed > 0:
                print(f"  Filtered {removed} master_weight tensors ({len(ckpt)} keys remaining)")
        sharded_weights.append(ckpt)
    return sharded_weights


def fix_norm_weights_from_pipeline(sharded_weights, pipe_transformer, expected_norm_size):
    unsharded_norms = {}
    for key, value in pipe_transformer.state_dict().items():
        if 'norm_k.weight' in key or 'norm_q.weight' in key:
            unsharded_norms[f"transformer.{key}"] = value.clone()
    if not unsharded_norms:
        return
    num_ranks = len(sharded_weights)
    norm_tp = unsharded_norms[list(unsharded_norms.keys())[0]].shape[0] // expected_norm_size
    fixed_count = 0
    for rank in range(num_ranks):
        ckpt = sharded_weights[rank]
        norm_rank = rank % norm_tp
        for norm_key, full_weight in unsharded_norms.items():
            if norm_key in ckpt and ckpt[norm_key].shape[0] != expected_norm_size:
                start = norm_rank * expected_norm_size
                ckpt[norm_key] = full_weight[start:start + expected_norm_size].to(ckpt[norm_key].dtype).clone()
                fixed_count += 1
    if fixed_count > 0:
        print(f"  Fixed {fixed_count} norm weights to size {expected_norm_size} (norm_tp={norm_tp})")


def load_duplicated_weights(model_path, world_size):
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)
    sharded_weights = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded_weights.append(ckpt)
    return sharded_weights


def unload_neuron_model(nxd_model, label="model"):
    """Thoroughly unload an NxDModel to release NeuronCore HBM resources.

    Simple del + gc.collect() is insufficient -- NRT resources hold HBM.
    This tears down per-rank SPMD models individually and forces GC.
    Reference: LTX-2.3 unload_neuron_model() pattern.
    """
    t0 = time.time()
    # Clear SPMD models (C++ NRT objects that hold HBM)
    if hasattr(nxd_model, "spmd_models"):
        for key in list(nxd_model.spmd_models.keys()):
            nxd_model.spmd_models[key] = None
        nxd_model.spmd_models.clear()
    # Clear weights
    if hasattr(nxd_model, "weights"):
        for i in range(len(nxd_model.weights)):
            nxd_model.weights[i] = {}
    # Clear states
    if hasattr(nxd_model, "states"):
        for i in range(len(nxd_model.states)):
            nxd_model.states[i] = {}
    # Clear reserved examples
    if hasattr(nxd_model, "reserved_example_inputs"):
        nxd_model.reserved_example_inputs.clear()
    if hasattr(nxd_model, "reserved_example_outputs"):
        nxd_model.reserved_example_outputs.clear()
    nxd_model.loaded_on_neuron = False
    del nxd_model
    gc.collect()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    time.sleep(2)  # Give NRT time to release resources
    print(f"  {label} unloaded from NeuronCores ({time.time() - t0:.1f}s)")


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    sharded_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
            sharded_checkpoints.append(ckpt)
    print(f"  Prepared {len(sharded_checkpoints)} checkpoints (TP={tp_degree}, CP={cp_degree})")
    return sharded_checkpoints


# ============================================================
# Phase 1: Text Encoding
# ============================================================
def phase_text_encoding_cpu(pipe, seqlen, prompt, negative_prompt):
    """Encode prompt using CPU text encoder."""
    print("Encoding prompt on CPU...")
    device = torch.device("cpu")
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=seqlen,
        device=device,
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}, negative: {negative_prompt_embeds.shape}")
    print(f"  Text encoding done in {time.time() - t0:.1f}s")
    return prompt_embeds, negative_prompt_embeds


def phase_text_encoding_neuron(pipe, compiled_models_dir, seqlen, prompt, negative_prompt):
    """Encode prompt using Neuron-compiled text encoder (load -> run -> unload).

    Uses InferenceTextEncoderWrapperV2 to wrap NxDModel and temporarily replaces
    pipe.text_encoder, then calls pipe.encode_prompt() which handles tokenization
    and attention mask application correctly.
    """
    te_path = f"{compiled_models_dir}/text_encoder"
    te_config = load_model_config(te_path)
    te_tp_degree = te_config["tp_degree"]
    te_world_size = te_config["world_size"]

    # Create wrapper
    text_encoder_wrapper = InferenceTextEncoderWrapperV2(
        torch.bfloat16, pipe.text_encoder, seqlen
    )

    # Load NxDModel
    print(f"Loading text encoder (TP={te_tp_degree}, world_size={te_world_size})...")
    t0 = time.time()
    te_nxd = NxDModel.load(os.path.join(te_path, "nxd_model.pt"))

    # Load weights (with CP expansion if needed)
    tp_checkpoints = load_sharded_weights(te_path, te_tp_degree)
    if te_world_size > te_tp_degree:
        cp_degree = te_world_size // te_tp_degree
        checkpoints = prepare_cp_checkpoints(tp_checkpoints, te_tp_degree, cp_degree)
    else:
        checkpoints = tp_checkpoints

    te_nxd.set_weights(checkpoints)
    te_nxd.to_neuron()
    print(f"  Text encoder loaded in {time.time() - t0:.1f}s")

    # Replace pipe's text encoder with NxDModel wrapper
    text_encoder_wrapper.t = te_nxd
    original_text_encoder = pipe.text_encoder
    pipe.text_encoder = text_encoder_wrapper

    # Run encoding via pipe.encode_prompt (handles tokenization + mask correctly)
    t_enc = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=seqlen,
        device=torch.device("cpu"),
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}, negative: {negative_prompt_embeds.shape}")
    print(f"  Text encoding done in {time.time() - t_enc:.1f}s (load: {t_enc - t0:.1f}s)")

    # Restore and clean up
    pipe.text_encoder = original_text_encoder
    unload_neuron_model(te_nxd, "Text encoder")
    del tp_checkpoints, checkpoints, text_encoder_wrapper
    gc.collect()

    return prompt_embeds, negative_prompt_embeds


def _run_text_encoding_subprocess(compiled_models_dir, seqlen, prompt, negative_prompt, pipe):
    """Run text encoding in a subprocess for clean HBM lifecycle."""
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="text_enc_")
    input_path = os.path.join(tmpdir, "input.pt")
    output_path = os.path.join(tmpdir, "output.pt")
    env_config_path = os.path.join(tmpdir, "env.json")

    env_config = {
        "NEURON_RT_NUM_CORES": os.environ.get("NEURON_RT_NUM_CORES", "8"),
        "NEURON_RT_VIRTUAL_CORE_SIZE": os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", "2"),
        "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0-7"),
        "NEURON_RT_INSPECT_ENABLE": "0",
        "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
        "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
        "NEURON_RT_PROFILING_MODE": "0",
    }
    with open(env_config_path, "w") as f:
        json.dump(env_config, f)

    # Tokenize in main process (no Neuron needed)
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length", max_length=seqlen,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )
    neg_text_inputs = pipe.tokenizer(
        negative_prompt, padding="max_length", max_length=seqlen,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )

    torch.save({
        "te_path": f"{compiled_models_dir}/text_encoder",
        "seqlen": seqlen,
        "prompt_input_ids": text_inputs.input_ids,
        "prompt_attention_mask": text_inputs.attention_mask,
        "neg_input_ids": neg_text_inputs.input_ids,
        "neg_attention_mask": neg_text_inputs.attention_mask,
    }, input_path)

    print(f"[Subprocess] Running text encoding in separate process...")
    t0 = time.time()

    cwd = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for k in list(env.keys()):
        if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
            del env[k]
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = cwd
    elif cwd not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = cwd + ":" + env["PYTHONPATH"]

    result = subprocess.run(
        [sys.executable, "-m", "neuron_wan2_2_t2v_a14b.text_encoder_worker",
         input_path, output_path, env_config_path],
        cwd=cwd, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Text encoding subprocess failed with code {result.returncode}")

    output_data = torch.load(output_path, weights_only=False)
    elapsed = time.time() - t0
    print(f"[Subprocess] Text encoding done in {elapsed:.1f}s (load: {output_data['load_time']:.1f}s, enc: {output_data['enc_time']:.1f}s)")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return output_data["prompt_embeds"], output_data["negative_prompt_embeds"]


def phase_text_encoding(pipe, compiled_models_dir, seqlen, prompt, negative_prompt,
                        use_neuron=True):
    """Encode prompt. Uses Neuron by default, CPU with --cpu_text_encoder."""
    print("\n" + "="*60)
    mode = "Neuron" if use_neuron else "CPU"
    print(f"PHASE 1: Text Encoding ({mode})")
    print("="*60)

    if use_neuron:
        return _run_text_encoding_subprocess(compiled_models_dir, seqlen, prompt, negative_prompt, pipe)
    else:
        return phase_text_encoding_cpu(pipe, seqlen, prompt, negative_prompt)


# ============================================================
# Phase 2: Denoising with MoE Transformer Swap (replace_weights)
# ============================================================
def load_and_prepare_weights(compiled_path, pipe_transformer, tp_degree, cp_degree, label="transformer"):
    """Load and prepare sharded+CP weights from a compiled transformer directory."""
    print(f"Loading weights for {label}...")
    tp_checkpoints = load_sharded_weights(compiled_path, tp_degree)

    # Detect and fix norm weights
    hidden_size = 5120
    ideal_norm_size = hidden_size // tp_degree
    expected_norm_size = None
    for key in tp_checkpoints[0]:
        if 'norm_q.weight' in key:
            actual_size = tp_checkpoints[0][key].shape[0]
            if actual_size != ideal_norm_size and actual_size != hidden_size:
                expected_norm_size = actual_size
            break
    if expected_norm_size is not None:
        fix_norm_weights_from_pipeline(tp_checkpoints, pipe_transformer, expected_norm_size)

    cp_checkpoints = prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree)
    return cp_checkpoints


def run_transformer_step(nxd_model, hidden_states, timestep, encoder_hidden_states,
                          rotary_emb_cos, rotary_emb_sin):
    """Run a single transformer forward pass."""
    if timestep is not None:
        if timestep.dim() > 1:
            timestep = timestep.flatten()[0:1]
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(torch.float32)

    output = nxd_model(
        hidden_states, timestep, encoder_hidden_states,
        rotary_emb_cos, rotary_emb_sin,
    )

    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _run_denoising_subprocess(phase_label, transformer_path, latents, prompt_embeds,
                                negative_prompt_embeds, timesteps, step_start, step_end,
                                guidance_scale, expand_timesteps, mask, scheduler_config,
                                scheduler_state):
    """Run a denoising phase in a subprocess for clean HBM lifecycle."""
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="denoise_")
    input_path = os.path.join(tmpdir, "input.pt")
    output_path = os.path.join(tmpdir, "output.pt")
    env_config_path = os.path.join(tmpdir, "env.json")

    # Save env config for subprocess
    env_config = {
        "NEURON_RT_NUM_CORES": os.environ.get("NEURON_RT_NUM_CORES", "32"),
        "NEURON_RT_VIRTUAL_CORE_SIZE": os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", "2"),
        "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0-31"),
        "NEURON_RT_INSPECT_ENABLE": "0",
        "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
        "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
        "NEURON_RT_PROFILING_MODE": "0",
    }
    with open(env_config_path, "w") as f:
        json.dump(env_config, f)

    # Save phase input data
    phase_data = {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "timesteps": timesteps,
        "step_start": step_start,
        "step_end": step_end,
        "guidance_scale": guidance_scale,
        "transformer_path": transformer_path,
        "expand_timesteps": expand_timesteps,
        "mask": mask,
        "scheduler_config": scheduler_config,
    }
    phase_data.update(scheduler_state)
    torch.save(phase_data, input_path)

    print(f"\n[Subprocess] Running {phase_label} (steps {step_start}-{step_end-1}) in separate process...")
    t0 = time.time()

    cwd = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Remove Neuron env vars so subprocess sets its own
    for k in list(env.keys()):
        if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
            del env[k]
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = cwd
    elif cwd not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = cwd + ":" + env["PYTHONPATH"]

    result = subprocess.run(
        [sys.executable, "-m", "neuron_wan2_2_t2v_a14b.denoise_worker",
         input_path, output_path, env_config_path],
        cwd=cwd, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Denoising subprocess for {phase_label} failed with code {result.returncode}")

    # Load output
    output_data = torch.load(output_path, weights_only=False)
    elapsed = time.time() - t0
    print(f"[Subprocess] {phase_label} done in {elapsed:.1f}s (load: {output_data['load_time']:.1f}s, denoise: {output_data['phase_time']:.1f}s)")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return output_data["latents"], output_data


def _run_denoising_combined_subprocess(transformer_1_path, transformer_2_path,
                                        latents, prompt_embeds, negative_prompt_embeds,
                                        timesteps, switch_idx, guidance_scale_high, guidance_scale_low,
                                        expand_timesteps, mask, scheduler_config, scheduler_state):
    """Run both MoE phases in a single subprocess using replace_weights()."""
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="denoise_combined_")
    input_path = os.path.join(tmpdir, "input.pt")
    output_path = os.path.join(tmpdir, "output.pt")
    env_config_path = os.path.join(tmpdir, "env.json")

    env_config = {
        "NEURON_RT_NUM_CORES": os.environ.get("NEURON_RT_NUM_CORES", "32"),
        "NEURON_RT_VIRTUAL_CORE_SIZE": os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", "2"),
        "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0-31"),
        "NEURON_RT_INSPECT_ENABLE": "0",
        "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
        "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
        "NEURON_RT_PROFILING_MODE": "0",
    }
    with open(env_config_path, "w") as f:
        json.dump(env_config, f)

    phase_data = {
        "mode": "combined",
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "timesteps": timesteps,
        "switch_idx": switch_idx,
        "guidance_scale_high": guidance_scale_high,
        "guidance_scale_low": guidance_scale_low,
        "transformer_1_path": transformer_1_path,
        "transformer_2_path": transformer_2_path,
        "expand_timesteps": expand_timesteps,
        "mask": mask,
        "scheduler_config": scheduler_config,
    }
    phase_data.update(scheduler_state)
    torch.save(phase_data, input_path)

    print(f"\n[Subprocess] Running combined MoE denoising (40 steps, replace_weights) in single process...")
    t0 = time.time()

    cwd = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for k in list(env.keys()):
        if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
            del env[k]
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = cwd
    elif cwd not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = cwd + ":" + env["PYTHONPATH"]

    result = subprocess.run(
        [sys.executable, "-m", "neuron_wan2_2_t2v_a14b.denoise_worker",
         input_path, output_path, env_config_path],
        cwd=cwd, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Combined denoising subprocess failed with code {result.returncode}")

    output_data = torch.load(output_path, weights_only=False)
    elapsed = time.time() - t0
    print(f"[Subprocess] Combined MoE done in {elapsed:.1f}s "
          f"(load: {output_data['load_time']:.1f}s, swap: {output_data.get('swap_time', 0):.1f}s, "
          f"phase1: {output_data.get('phase1_time', 0):.1f}s, phase2: {output_data.get('phase2_time', 0):.1f}s)")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return output_data["latents"], output_data


def phase_denoising(pipe, compiled_models_dir, prompt_embeds, negative_prompt_embeds,
                     args, generator=None):
    """Run the denoising loop with MoE transformer swap.

    Each expert phase runs in a subprocess for clean HBM lifecycle.
    Subprocess exit guarantees full NRT resource release between phases.
    """
    print("\n" + "="*60)
    print("PHASE 2: Denoising (MoE)")
    print("="*60)

    DTYPE = torch.bfloat16
    device = torch.device("cpu")

    prompt_embeds = prompt_embeds.to(DTYPE)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)

    # Prepare scheduler and timesteps
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latents
    in_channels = pipe.transformer.config.in_channels if pipe.transformer is not None else 16
    latents = pipe.prepare_latents(
        1, in_channels, args.height, args.width, args.num_frames,
        torch.float32, device, generator, None,
    )
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    # MoE boundary
    boundary_timestep = 0.875 * 1000  # 875.0
    guidance_scale_high = args.guidance_scale    # high-noise expert (transformer_1)
    guidance_scale_low = args.guidance_scale_2   # low-noise expert (transformer_2)

    # Determine switch point
    switch_idx = None
    for i, t in enumerate(timesteps):
        if t < boundary_timestep:
            switch_idx = i
            break
    if switch_idx is None:
        switch_idx = len(timesteps)

    print(f"Total steps: {len(timesteps)}, switch at step {switch_idx} (boundary={boundary_timestep})")
    print(f"  transformer_1 (high-noise): steps 0-{switch_idx-1} ({switch_idx} steps)")
    print(f"  transformer_2 (low-noise):  steps {switch_idx}-{len(timesteps)-1} ({len(timesteps) - switch_idx} steps)")

    transformer_1_path = f"{compiled_models_dir}/transformer"
    transformer_2_path = f"{compiled_models_dir}/transformer_2"

    # Load config from transformer_1
    config = load_model_config(transformer_1_path)
    world_size = config["world_size"]
    hbm_tight = True  # Always use subprocess for clean HBM lifecycle

    if hbm_tight:
        # ============================================================
        # SUBPROCESS MODE: Run each phase in a separate process to
        # ensure clean HBM lifecycle (required for 720P).
        # ============================================================
        print(f"\nUsing subprocess mode for clean HBM (world_size={world_size})")

        scheduler_config = dict(pipe.scheduler.config)
        scheduler_state_initial = {
            "scheduler_order_list": getattr(pipe.scheduler, 'order_list', None),
            "scheduler_model_outputs": getattr(pipe.scheduler, 'model_outputs', None),
            "scheduler_timestep_list": getattr(pipe.scheduler, 'timestep_list', None),
            "scheduler_lower_order_nums": getattr(pipe.scheduler, 'lower_order_nums', None),
            "scheduler_sample": getattr(pipe.scheduler, 'sample', None),
        }

        # Combined MoE: load T1, run, replace_weights T2, run, exit
        # Saves ~70s by avoiding second NxDModel.load()
        latents, out2 = _run_denoising_combined_subprocess(
            transformer_1_path, transformer_2_path,
            latents, prompt_embeds, negative_prompt_embeds,
            timesteps, switch_idx, guidance_scale_high, guidance_scale_low,
            pipe.config.expand_timesteps, mask, scheduler_config,
            scheduler_state_initial,
        )

    else:
        # ============================================================
        # IN-PROCESS MODE: Use replace_weights() for fast MoE swap.
        # Works for 480P where NEFF has HBM headroom.
        # ============================================================
        tp_degree = config["tp_degree"]
        cp_degree = config["cp_degree"]

        # Load RoPE (same for both transformers - same architecture)
        rope_cache = torch.load(os.path.join(transformer_1_path, "rope_cache.pt"))
        rotary_emb_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
        rotary_emb_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)

        # Prepare weights for both transformers
        t1_weights = load_and_prepare_weights(
            transformer_1_path, pipe.transformer, tp_degree, cp_degree, "transformer_1"
        )
        t2_weights = load_and_prepare_weights(
            transformer_2_path, pipe.transformer_2, tp_degree, cp_degree, "transformer_2"
        )

        # Load ONE NxDModel (from transformer_1's NEFF)
        nxd_model_path = os.path.join(transformer_1_path, "nxd_model.pt")
        nxd_model = NxDModel.load(nxd_model_path, start_rank=0, local_ranks_size=world_size)

        # Initialize with transformer_1 weights
        print(f"\nLoading NxDModel with transformer_1 weights (TP={tp_degree}, CP={cp_degree})...")
        nxd_model.set_weights(t1_weights)
        t0 = time.time()
        nxd_model.to_neuron()
        print(f"  NxDModel loaded to NeuronCores in {time.time() - t0:.1f}s")

        # ---- Run high-noise steps with transformer_1 weights ----
        print(f"\nRunning {switch_idx} high-noise denoising steps (transformer_1)...")
        t0 = time.time()
        for i in range(switch_idx):
            t = timesteps[i]
            latent_input = latents.to(DTYPE)

            if pipe.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                ts = t.expand(latents.shape[0])

            noise_pred = run_transformer_step(nxd_model, latent_input, ts, prompt_embeds,
                                               rotary_emb_cos, rotary_emb_sin)
            noise_uncond = run_transformer_step(nxd_model, latent_input, ts, negative_prompt_embeds,
                                                 rotary_emb_cos, rotary_emb_sin)
            noise_pred = noise_uncond.float() + guidance_scale_high * (noise_pred.float() - noise_uncond.float())
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            elapsed = time.time() - t0
            print(f"  Step {i+1}/{switch_idx} (t={t.item():.0f}) - {elapsed:.1f}s elapsed")

        print(f"High-noise phase done. {switch_idx} steps in {time.time() - t0:.1f}s")

        # ---- Swap weights to transformer_2 using replace_weights() ----
        remaining_steps = len(timesteps) - switch_idx
        if remaining_steps > 0:
            print(f"\nSwapping weights to transformer_2 via replace_weights()...")
            t_swap = time.time()
            nxd_model.replace_weights(t2_weights)
            swap_time = time.time() - t_swap
            print(f"  Weight swap completed in {swap_time:.1f}s")

            del t1_weights
            gc.collect()

            # ---- Run low-noise steps with transformer_2 weights ----
            print(f"\nRunning {remaining_steps} low-noise denoising steps (transformer_2)...")
            t0 = time.time()
            for i in range(switch_idx, len(timesteps)):
                t = timesteps[i]
                latent_input = latents.to(DTYPE)

                if pipe.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    ts = t.expand(latents.shape[0])

                noise_pred = run_transformer_step(nxd_model, latent_input, ts, prompt_embeds,
                                                   rotary_emb_cos, rotary_emb_sin)
                noise_uncond = run_transformer_step(nxd_model, latent_input, ts, negative_prompt_embeds,
                                                     rotary_emb_cos, rotary_emb_sin)
                noise_pred = noise_uncond.float() + guidance_scale_low * (noise_pred.float() - noise_uncond.float())
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                step_num = i - switch_idx + 1
                if step_num % 10 == 0 or i == len(timesteps) - 1:
                    elapsed = time.time() - t0
                    print(f"  Step {step_num}/{remaining_steps} (t={t.item():.0f}) - {elapsed:.1f}s elapsed")

            print(f"Low-noise phase done. {remaining_steps} steps in {time.time() - t0:.1f}s")

        # Cleanup
        unload_neuron_model(nxd_model, "Transformer")
        del t2_weights
        gc.collect()

    return latents


def _run_vae_decode_subprocess(pipe, compiled_models_dir, latents, num_frames=81):
    """Run VAE decode (post_quant_conv + decoder) in a subprocess for clean HBM."""
    import subprocess
    import tempfile

    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (Neuron subprocess)")
    print("="*60)

    # Denormalize latents on CPU (no Neuron needed)
    vae_config = pipe.vae.config
    latents = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(vae_config.latents_mean)
        .view(1, vae_config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(
        1, vae_config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"Denormalized latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    tmpdir = tempfile.mkdtemp(prefix="vae_dec_")
    input_path = os.path.join(tmpdir, "input.pt")
    output_path = os.path.join(tmpdir, "output.pt")
    env_config_path = os.path.join(tmpdir, "env.json")

    env_config = {
        "NEURON_RT_NUM_CORES": os.environ.get("NEURON_RT_NUM_CORES", "8"),
        "NEURON_RT_VIRTUAL_CORE_SIZE": os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", "2"),
        "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0-7"),
        "NEURON_RT_INSPECT_ENABLE": "0",
        "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
        "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
        "NEURON_RT_PROFILING_MODE": "0",
    }
    with open(env_config_path, "w") as f:
        json.dump(env_config, f)

    torch.save({
        "compiled_models_dir": compiled_models_dir,
        "latents_f32": latents,
        "z_bf16": latents.to(torch.bfloat16),
        "num_frames": num_frames,
    }, input_path)

    print(f"[Subprocess] Running VAE decode in separate process...")
    t0 = time.time()

    cwd = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for k in list(env.keys()):
        if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
            del env[k]
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = cwd
    elif cwd not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = cwd + ":" + env["PYTHONPATH"]

    result = subprocess.run(
        [sys.executable, "-m", "neuron_wan2_2_t2v_a14b.decoder_worker",
         input_path, output_path, env_config_path],
        cwd=cwd, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VAE decode subprocess failed with code {result.returncode}")

    output_data = torch.load(output_path, weights_only=False)
    elapsed = time.time() - t0
    print(f"[Subprocess] VAE decode done in {elapsed:.1f}s (decode: {output_data['decode_time']:.1f}s)")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    video = output_data["video"].numpy()
    print(f"Output video: shape={video.shape}, dtype={video.dtype}")
    return video


# ============================================================
# Phase 3: VAE Decoding (Neuron)
# ============================================================
def phase_vae_decode(pipe, compiled_models_dir, latents, num_frames=81):
    """Decode latents using compiled Neuron VAE (post_quant_conv + chunked decoder).

    Pipeline:
    1. Denormalize latents using VAE config (latents_mean/latents_std)
    2. Run Neuron post_quant_conv on full latent volume (float32, all 21 frames)
    3. Run Neuron decoder in chunks of decoder_frames (default 2) latent frames
       Each chunk produces decoder_frames x 4 video frames (4x temporal upsample)
    4. Concatenate and trim to expected frame count

    Decoder modes (auto-detected based on compiled model):
    - Rolling cache (decoder_rolling/): feat_cache carried between chunks as I/O.
      Flicker-free but ~1.8GB extra transfer per chunk.
    - NoCache (decoder_nocache/): feat_cache as zero buffers. Fast but flickering.
    """
    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (Neuron)")
    print("="*60)

    # Denormalize latents using pipe's VAE config (already loaded in memory)
    vae_config = pipe.vae.config
    latents = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(vae_config.latents_mean)
        .view(1, vae_config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(
        1, vae_config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"Denormalized latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # ---- Load and run post_quant_conv ----
    pqc_path = f"{compiled_models_dir}/post_quant_conv"
    pqc_config = load_model_config(pqc_path)
    pqc_world_size = pqc_config["world_size"]

    print(f"\nLoading post_quant_conv (world_size={pqc_world_size})...")
    t0 = time.time()
    pqc_nxd = NxDModel.load(
        os.path.join(pqc_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=pqc_world_size,
    )
    pqc_weights = load_duplicated_weights(pqc_path, pqc_world_size)
    pqc_nxd.set_weights(pqc_weights)
    pqc_nxd.to_neuron()
    print(f"  post_quant_conv loaded in {time.time() - t0:.1f}s")

    print(f"Running post_quant_conv on {latents.shape}...")
    t0 = time.time()
    z = pqc_nxd(latents)
    if isinstance(z, (tuple, list)):
        z = z[0]
    z = z.to(torch.float32)
    print(f"  post_quant_conv done in {time.time() - t0:.1f}s, output: {z.shape}")

    unload_neuron_model(pqc_nxd, "post_quant_conv")
    del pqc_weights
    gc.collect()

    # ---- Load and run decoder ----
    # Try rolling cache first (flicker-free), fall back to nocache
    rolling_path = f"{compiled_models_dir}/decoder_rolling"
    nocache_path = f"{compiled_models_dir}/decoder_nocache"

    if os.path.exists(os.path.join(rolling_path, "nxd_model.pt")):
        decoder_path = rolling_path
        is_rolling = True
    else:
        decoder_path = nocache_path
        is_rolling = False

    decoder_config = load_model_config(decoder_path)
    decoder_world_size = decoder_config["world_size"]
    decoder_frames = decoder_config.get("decoder_frames", 2)
    mode_str = "rolling cache" if is_rolling else "nocache"

    print(f"\nLoading decoder [{mode_str}] (world_size={decoder_world_size}, compiled_frames={decoder_frames})...")
    t0 = time.time()
    decoder_nxd = NxDModel.load(
        os.path.join(decoder_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=decoder_world_size,
    )
    decoder_weights = load_duplicated_weights(decoder_path, decoder_world_size)
    decoder_nxd.set_weights(decoder_weights)
    decoder_nxd.to_neuron()
    print(f"  Decoder loaded in {time.time() - t0:.1f}s")

    z_bf16 = z.to(torch.bfloat16)
    num_latent_frames = z_bf16.shape[2]
    decoded_frames = []

    num_chunks = (num_latent_frames + decoder_frames - 1) // decoder_frames
    print(f"Decoding {num_latent_frames} latent frames in {num_chunks} chunks of {decoder_frames} [{mode_str}]...")
    decode_start = time.time()

    # Initialize rolling cache (zero tensors, same shapes as compiled)
    if is_rolling:
        from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes
        latent_h, latent_w = z_bf16.shape[3], z_bf16.shape[4]
        cache_shapes = get_feat_cache_shapes(1, latent_h, latent_w)
        caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]
        print(f"  Rolling cache initialized: {len(caches)} tensors")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * decoder_frames
        end = min(start + decoder_frames, num_latent_frames)
        chunk = z_bf16[:, :, start:end, :, :]

        # Pad last chunk if needed
        if chunk.shape[2] < decoder_frames:
            pad_frames = decoder_frames - chunk.shape[2]
            padding = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
            chunk = torch.cat([chunk, padding], dim=2)

        if is_rolling:
            # Rolling mode: pass cache as inputs, get updated cache as outputs
            results = decoder_nxd(chunk, *caches)
            if isinstance(results, (tuple, list)):
                output = results[0]
                caches = [r.to(torch.bfloat16) for r in results[1:1 + len(cache_shapes)]]
            else:
                output = results
        else:
            # NoCache mode: just pass x
            output = decoder_nxd(chunk)
            if isinstance(output, (list, tuple)):
                output = output[0]

        output = output.to(torch.float32)

        # Each chunk of decoder_frames latent frames -> decoder_frames * 4 video frames
        actual_latent = end - start
        video_frames_from_chunk = actual_latent * 4
        output = output[:, :, :video_frames_from_chunk]
        decoded_frames.append(output)

        elapsed = time.time() - decode_start
        print(f"  Chunk {chunk_idx+1}/{num_chunks}: latent [{start}:{end}] -> {output.shape[2]} video frames ({elapsed:.1f}s)")

    video = torch.cat(decoded_frames, dim=2)  # [B, C, total_video_frames, H, W]
    decode_time = time.time() - decode_start

    # Trim to expected number of video frames
    if video.shape[2] > num_frames:
        print(f"  Trimming {video.shape[2]} -> {num_frames} video frames")
        video = video[:, :, :num_frames]

    print(f"  Total decode time: {decode_time:.1f}s, output: {video.shape} [{mode_str}]")

    unload_neuron_model(decoder_nxd, "Decoder")
    del decoder_weights, z, z_bf16, decoded_frames
    gc.collect()

    # Post-processing: [B, C, F, H, W] -> numpy [F, H, W, C], float [0,1]
    # NOTE: export_to_video expects float [0,1] ndarray and does *255 internally.
    # Do NOT pass uint8, otherwise it double-multiplies by 255!
    video = video[0]  # [C, F, H, W]
    video = video.permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
    video = ((video + 1.0) / 2.0).clip(0, 1)
    print(f"Output video: shape={video.shape}, dtype={video.dtype}")

    return video


# ============================================================
# Phase 3 (Neuron Tiled): VAE Decoding with 480P tile patches
# ============================================================
def phase_vae_decode_neuron_tiled(pipe, compiled_models_dir, latents, num_frames=81):
    """Decode latents using tiled Neuron VAE decode with 480P decoder patches.

    For 720P (90x160 latent), tiles the volume into overlapping 480P-sized patches
    (60x104 latent), decodes each on Neuron using the compiled 480P rolling-cache
    decoder, and blends tiles with linear ramp weighting in overlap zones.

    post_quant_conv runs on CPU for the full volume (Conv3d(16,16,1) is negligible).
    """
    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (Neuron Tiled)")
    print("="*60)

    import torch.nn.functional as F

    # Lazy Neuron import (only if not already imported by top-level code)
    if 'torch_neuronx' not in sys.modules:
        decoder_ws = 8
        decoder_num_cores = decoder_ws
        # Derive core range from current VISIBLE_CORES (set by top-level init or user)
        visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "0-63")
        core_start = int(visible.split("-")[0])
        os.environ["NEURON_RT_NUM_CORES"] = str(decoder_num_cores)
        os.environ["NEURON_RT_VISIBLE_CORES"] = f"{core_start}-{core_start + decoder_num_cores - 1}"
        print(f"Initializing Neuron for tiled decoder: NUM_CORES={decoder_num_cores}, cores={core_start}-{core_start + decoder_num_cores - 1}")

    import torch_neuronx  # noqa: F811 — may re-import (safe, Python caches modules)
    from neuronx_distributed import NxDModel  # noqa: F811
    from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes

    # Denormalize latents on CPU
    vae_config = pipe.vae.config
    latents = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(vae_config.latents_mean)
        .view(1, vae_config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(
        1, vae_config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"Denormalized latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Run post_quant_conv on CPU for full volume (Conv3d(16,16,1) is negligible)
    print("Running post_quant_conv on CPU...")
    t0 = time.time()
    pipe.vae.post_quant_conv.to(torch.float32)
    with torch.no_grad():
        z = pipe.vae.post_quant_conv(latents)
    print(f"  post_quant_conv done in {time.time() - t0:.1f}s, output: {z.shape}")

    z_bf16 = z.to(torch.bfloat16)
    del z, latents
    gc.collect()

    B, C, T, H_lat, W_lat = z_bf16.shape

    # Tile parameters (480P decoder compiled for 60x104 latent)
    tile_h, tile_w = 60, 104
    overlap_h = tile_h // 4  # 15
    overlap_w = tile_w // 4  # 26
    stride_h = tile_h - overlap_h  # 45
    stride_w = tile_w - overlap_w  # 78

    # Compute tile positions (stop when current tile covers the full dimension)
    h_starts = []
    h = 0
    while h < H_lat:
        h_starts.append(h)
        if h + tile_h >= H_lat:
            break
        h += stride_h
    w_starts = []
    w = 0
    while w < W_lat:
        w_starts.append(w)
        if w + tile_w >= W_lat:
            break
        w += stride_w

    n_tiles = len(h_starts) * len(w_starts)
    print(f"Tiling: latent {H_lat}x{W_lat} -> {len(h_starts)}x{len(w_starts)} = {n_tiles} tiles")
    print(f"  Tile size: {tile_h}x{tile_w}, stride: {stride_h}x{stride_w}, overlap: {overlap_h}x{overlap_w}")
    for hi, hs in enumerate(h_starts):
        for wi, ws in enumerate(w_starts):
            ah = min(tile_h, H_lat - hs)
            aw = min(tile_w, W_lat - ws)
            print(f"  Tile ({hi},{wi}): latent [{hs}:{hs+ah}, {ws}:{ws+aw}] (actual {ah}x{aw})")

    # Load 480P decoder
    decoder_path = f"{compiled_models_dir}/decoder_rolling_480p"
    decoder_config = load_model_config(decoder_path)
    decoder_world_size = decoder_config["world_size"]
    decoder_frames = decoder_config.get("decoder_frames", 2)

    print(f"\nLoading 480P decoder (world_size={decoder_world_size}, frames={decoder_frames})...")
    t0 = time.time()
    decoder_nxd = NxDModel.load(
        os.path.join(decoder_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=decoder_world_size,
    )
    decoder_weights = load_duplicated_weights(decoder_path, decoder_world_size)
    decoder_nxd.set_weights(decoder_weights)
    decoder_nxd.to_neuron()
    load_time = time.time() - t0
    print(f"  Decoder loaded in {load_time:.1f}s")

    # Pixel-space dimensions
    H_pix = H_lat * 8
    W_pix = W_lat * 8
    overlap_h_pix = overlap_h * 8
    overlap_w_pix = overlap_w * 8

    # Accumulation buffers (float32 for blending precision)
    num_latent_frames = T
    num_video_frames = num_frames
    output_acc = torch.zeros(3, num_video_frames, H_pix, W_pix, dtype=torch.float32)
    weight_acc = torch.zeros(H_pix, W_pix, dtype=torch.float32)

    num_chunks = (num_latent_frames + decoder_frames - 1) // decoder_frames
    decode_start = time.time()

    for hi, h_start in enumerate(h_starts):
        for wi, w_start in enumerate(w_starts):
            tile_idx = hi * len(w_starts) + wi
            actual_h = min(tile_h, H_lat - h_start)
            actual_w = min(tile_w, W_lat - w_start)

            # Extract spatial tile
            z_tile = z_bf16[:, :, :, h_start:h_start+actual_h, w_start:w_start+actual_w]

            # Pad to tile size if edge tile
            if actual_h < tile_h or actual_w < tile_w:
                z_tile = F.pad(z_tile, (0, tile_w - actual_w, 0, tile_h - actual_h))

            # Initialize rolling cache for this tile
            cache_shapes = get_feat_cache_shapes(1, tile_h, tile_w)
            caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]

            # Decode temporal chunks
            tile_frames = []
            tile_start = time.time()
            for chunk_idx in range(num_chunks):
                t_start = chunk_idx * decoder_frames
                t_end = min(t_start + decoder_frames, num_latent_frames)
                chunk = z_tile[:, :, t_start:t_end, :, :]

                # Pad last temporal chunk if needed
                if chunk.shape[2] < decoder_frames:
                    pad_t = decoder_frames - chunk.shape[2]
                    padding = chunk[:, :, -1:, :, :].expand(-1, -1, pad_t, -1, -1)
                    chunk = torch.cat([chunk, padding], dim=2)

                # Run decoder
                results = decoder_nxd(chunk, *caches)
                if isinstance(results, (tuple, list)):
                    output = results[0]
                    caches = [r.to(torch.bfloat16) for r in results[1:1 + len(cache_shapes)]]
                else:
                    output = results

                output = output.to(torch.float32)

                # Trim temporal padding
                actual_t = t_end - t_start
                video_frames_from_chunk = actual_t * 4
                output = output[:, :, :video_frames_from_chunk]
                tile_frames.append(output)

            tile_video = torch.cat(tile_frames, dim=2)  # [1, 3, F_total, tile_h*8, tile_w*8]

            # Trim to expected frame count
            if tile_video.shape[2] > num_video_frames:
                tile_video = tile_video[:, :, :num_video_frames]

            # Trim spatial padding from decoded output
            actual_h_pix = actual_h * 8
            actual_w_pix = actual_w * 8
            tile_video = tile_video[:, :, :, :actual_h_pix, :actual_w_pix]

            # Compute blending weight (linear ramps in overlap zones)
            h_weight = torch.ones(actual_h_pix)
            w_weight = torch.ones(actual_w_pix)

            if hi > 0:  # has tile above -> ramp at start
                ramp = min(overlap_h_pix, actual_h_pix)
                h_weight[:ramp] = torch.linspace(0, 1, ramp + 2)[1:-1]
            if hi < len(h_starts) - 1:  # has tile below -> ramp at end
                ramp = min(overlap_h_pix, actual_h_pix)
                h_weight[-ramp:] = torch.linspace(1, 0, ramp + 2)[1:-1]
            if wi > 0:  # has tile left -> ramp at start
                ramp = min(overlap_w_pix, actual_w_pix)
                w_weight[:ramp] = torch.linspace(0, 1, ramp + 2)[1:-1]
            if wi < len(w_starts) - 1:  # has tile right -> ramp at end
                ramp = min(overlap_w_pix, actual_w_pix)
                w_weight[-ramp:] = torch.linspace(1, 0, ramp + 2)[1:-1]

            weight_2d = h_weight.unsqueeze(1) * w_weight.unsqueeze(0)  # [ah_pix, aw_pix]

            # Accumulate weighted tile into output
            h_pix = h_start * 8
            w_pix = w_start * 8
            output_acc[:, :, h_pix:h_pix+actual_h_pix, w_pix:w_pix+actual_w_pix] += (
                tile_video[0] * weight_2d.unsqueeze(0).unsqueeze(0)
            )
            weight_acc[h_pix:h_pix+actual_h_pix, w_pix:w_pix+actual_w_pix] += weight_2d

            tile_time = time.time() - tile_start
            elapsed = time.time() - decode_start
            print(f"  Tile ({hi},{wi}) done: {num_chunks} chunks, {tile_time:.1f}s ({elapsed:.1f}s total)")

            del tile_video, tile_frames, z_tile, caches
            gc.collect()

    decode_time = time.time() - decode_start

    # Normalize by accumulated weights
    video = output_acc / weight_acc.unsqueeze(0).unsqueeze(0).clamp(min=1e-6)

    print(f"  Total tiled decode time: {decode_time:.1f}s (load: {load_time:.1f}s)")
    print(f"  Output: {list(video.shape)}")

    del decoder_nxd, decoder_weights, z_bf16, output_acc, weight_acc
    gc.collect()

    # Post-processing: [C, F, H, W] -> numpy [F, H, W, C], float [0,1]
    video = video.permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
    video = ((video + 1.0) / 2.0).clip(0, 1)
    print(f"Output video: shape={video.shape}, dtype={video.dtype}")

    return video


# ============================================================
# Phase 3 (CPU fallback): VAE Decoding on CPU
# ============================================================
def phase_vae_decode_cpu(pipe, latents, num_frames=81):
    """Decode latents using CPU VAE (fallback when Neuron decoder not available).

    Uses pipe.vae.decode() which handles post_quant_conv + decoder internally.
    Slower than Neuron but works for any resolution without compilation.
    """
    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (CPU fallback)")
    print("="*60)

    t0 = time.time()
    latents = latents.to(torch.float32)

    # Denormalize latents (same as Neuron path and diffusers pipeline)
    vae_config = pipe.vae.config
    latents_mean = (
        torch.tensor(vae_config.latents_mean)
        .view(1, vae_config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(
        1, vae_config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"Denormalized latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Decode on CPU using pipe.vae
    print("Running VAE decode on CPU (this may take a while for high resolutions)...")
    pipe.vae.to(torch.float32)
    with torch.no_grad():
        video = pipe.vae.decode(latents).sample
    decode_time = time.time() - t0

    # Trim to expected frame count
    if video.shape[2] > num_frames:
        print(f"  Trimming {video.shape[2]} -> {num_frames} video frames")
        video = video[:, :, :num_frames]

    print(f"  CPU decode time: {decode_time:.1f}s, output: {video.shape}")

    # Post-processing: [B, C, F, H, W] -> numpy [F, H, W, C], float [0,1]
    video = video[0]  # [C, F, H, W]
    video = video.permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
    video = ((video + 1.0) / 2.0).clip(0, 1)
    print(f"Output video: shape={video.shape}, dtype={video.dtype}")

    return video


# ============================================================
# Main
# ============================================================
DEFAULT_COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models_t2v_a14b"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"
SEED = 42


def main(args):
    total_start = time.time()
    set_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Load base pipeline (CPU)
    print("Loading base pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    compiled_models_dir = args.compiled_models_dir
    seqlen = args.max_sequence_length

    inference_start = time.time()

    # Phase 1: Text Encoding
    t1_start = time.time()
    prompt_embeds, negative_prompt_embeds = phase_text_encoding(
        pipe, compiled_models_dir, seqlen, args.prompt, args.negative_prompt,
        use_neuron=not args.cpu_text_encoder,
    )
    t1_time = time.time() - t1_start

    # Phase 2: Denoising with MoE swap
    print("\nStarting denoising...")
    generator = torch.Generator().manual_seed(SEED)
    t2_start = time.time()
    latents = phase_denoising(
        pipe, compiled_models_dir, prompt_embeds, negative_prompt_embeds, args, generator
    )
    t2_time = time.time() - t2_start
    print(f"\nTotal denoising time: {t2_time:.2f}s")
    print(f"Per step (including swap): {t2_time / args.num_inference_steps:.3f}s")

    # Save latents for debugging
    torch.save(latents, "debug_latents.pt")
    print(f"Saved debug_latents.pt: shape={latents.shape}, dtype={latents.dtype}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Phase 3: VAE Decode (Neuron or CPU fallback)
    t3_start = time.time()
    # Auto-detect decoder mode:
    # 1. Full-res Neuron decoder (480P in-process, requires torch_neuronx at top level)
    # 2. Tiled Neuron decoder (720P: 480P patches, lazy Neuron import)
    # 3. CPU fallback
    has_neuron_decoder = (
        not args.cpu_vae_decoder
        and (
            os.path.exists(os.path.join(compiled_models_dir, "decoder_rolling", "nxd_model.pt"))
            or os.path.exists(os.path.join(compiled_models_dir, "decoder_nocache", "nxd_model.pt"))
        )
    )
    has_tiled_decoder = (
        not args.cpu_vae_decoder
        and not has_neuron_decoder
        and os.path.exists(os.path.join(compiled_models_dir, "decoder_rolling_480p", "nxd_model.pt"))
    )
    if has_neuron_decoder:
        video = _run_vae_decode_subprocess(pipe, compiled_models_dir, latents, num_frames=args.num_frames)
    elif has_tiled_decoder:
        video = phase_vae_decode_neuron_tiled(pipe, compiled_models_dir, latents, num_frames=args.num_frames)
    else:
        if not args.cpu_vae_decoder:
            print("\nNo compiled Neuron decoder found, using CPU fallback.")
        video = phase_vae_decode_cpu(pipe, latents, num_frames=args.num_frames)
    t3_time = time.time() - t3_start

    inference_time = time.time() - inference_start

    # Export video
    output_path = args.output
    frames = [video[i] for i in range(video.shape[0])]
    print(f"Exporting {len(frames)} frames, shape={frames[0].shape}, dtype={frames[0].dtype}")
    export_to_video(frames, output_path, fps=16)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Video saved to: {output_path} ({len(frames)} frames)")
    print(f"{'='*60}")
    print(f"  Phase 1 - Text Encoding:  {t1_time:.1f}s")
    print(f"  Phase 2 - Denoising:      {t2_time:.1f}s")
    print(f"  Phase 3 - VAE Decode:     {t3_time:.1f}s")
    print(f"  Inference time:           {inference_time:.1f}s")
    print(f"  Total (incl. pipeline):   {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 T2V-A14B Inference with MoE Transformers")
    parser.add_argument("--compiled_models_dir", type=str, default=DEFAULT_COMPILED_MODELS_DIR)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="High-noise guidance scale (transformer_1)")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0,
                        help="Low-noise guidance scale (transformer_2)")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--output", type=str, default="output_t2v_a14b.mp4")
    parser.add_argument("--cpu_text_encoder", action="store_true",
                        help="Use CPU text encoder instead of Neuron (slower but more stable)")
    parser.add_argument("--cpu_vae_decoder", action="store_true",
                        help="Force CPU VAE decoder (auto-detected if no Neuron decoder compiled)")
    args = parser.parse_args()

    main(args)
