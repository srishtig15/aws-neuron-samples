"""
Subprocess worker for running denoising phases on Neuron.

Supports two modes:
1. Single phase: load one transformer, run steps, exit
2. Combined MoE: load transformer_1, run high-noise steps, replace_weights
   with transformer_2, run low-noise steps, exit. Saves ~70s by avoiding
   a second NxDModel.load().

Usage (called programmatically):
    python -m neuron_wan2_2_t2v_a14b.denoise_worker <input.pt> <output.pt> [env.json]
"""
import json
import os
import sys
import time

# Set env vars before any Neuron imports
input_path = sys.argv[1]
output_path = sys.argv[2]

# Load config to set env vars before importing torch_neuronx
config_path = sys.argv[3] if len(sys.argv) > 3 else None
if config_path:
    with open(config_path) as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        os.environ[k] = str(v)

import gc
import torch
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file


def load_sharded_weights(model_path, tp_degree):
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        sharded_weights.append(ckpt)
    return sharded_weights


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    sharded_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
            sharded_checkpoints.append(ckpt)
    return sharded_checkpoints


def run_transformer_step(nxd_model, hidden_states, timestep, encoder_hidden_states,
                          rotary_emb_cos, rotary_emb_sin):
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


def run_transformer_step_batched(nxd_model, hidden_states, timestep,
                                  prompt_embeds, negative_prompt_embeds,
                                  rotary_emb_cos, rotary_emb_sin):
    """Run batched CFG: concat prompt+negative into batch=2, single forward pass."""
    if timestep is not None:
        if timestep.dim() > 1:
            timestep = timestep.flatten()[0:1]
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(torch.float32)
        timestep = timestep.expand(2)

    hidden_states_batched = hidden_states.expand(2, *hidden_states.shape[1:]).contiguous()
    encoder_hidden_states_batched = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    rotary_emb_cos_batched = rotary_emb_cos.expand(2, *rotary_emb_cos.shape[1:]).contiguous()
    rotary_emb_sin_batched = rotary_emb_sin.expand(2, *rotary_emb_sin.shape[1:]).contiguous()

    output = nxd_model(
        hidden_states_batched, timestep, encoder_hidden_states_batched,
        rotary_emb_cos_batched, rotary_emb_sin_batched,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]

    noise_pred = output[0:1]
    noise_uncond = output[1:2]
    return noise_pred, noise_uncond


def run_phase_steps(nxd_model, latents, prompt_embeds, negative_prompt_embeds,
                     timesteps, step_start, step_end, guidance_scale,
                     expand_timesteps, mask, scheduler, rotary_emb_cos, rotary_emb_sin,
                     compiled_batch_size=1):
    """Run denoising steps for one phase."""
    DTYPE = torch.bfloat16
    num_steps = step_end - step_start

    t0 = time.time()
    for i in range(step_start, step_end):
        t = timesteps[i]
        latent_input = latents.to(DTYPE)

        if expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            ts = t.expand(latents.shape[0])

        if compiled_batch_size >= 2:
            noise_pred, noise_uncond = run_transformer_step_batched(
                nxd_model, latent_input, ts,
                prompt_embeds, negative_prompt_embeds,
                rotary_emb_cos, rotary_emb_sin,
            )
        else:
            noise_pred = run_transformer_step(nxd_model, latent_input, ts, prompt_embeds,
                                               rotary_emb_cos, rotary_emb_sin)
            noise_uncond = run_transformer_step(nxd_model, latent_input, ts, negative_prompt_embeds,
                                                 rotary_emb_cos, rotary_emb_sin)

        noise_pred = noise_uncond.float() + guidance_scale * (noise_pred.float() - noise_uncond.float())
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        step_num = i - step_start + 1
        if step_num % 5 == 0 or i == step_end - 1 or step_num == 1:
            elapsed = time.time() - t0
            print(f"  Step {step_num}/{num_steps} (t={t.item():.0f}) - {elapsed:.1f}s elapsed")

    phase_time = time.time() - t0
    return latents, phase_time


def main():
    t_total = time.time()

    data = torch.load(input_path, weights_only=False)
    mode = data.get("mode", "single")  # "single" or "combined"

    if mode == "combined":
        main_combined(data)
    else:
        main_single(data)


def main_single(data):
    """Original single-phase mode."""
    t_total = time.time()

    latents = data["latents"]
    prompt_embeds = data["prompt_embeds"]
    negative_prompt_embeds = data["negative_prompt_embeds"]
    timesteps = data["timesteps"]
    step_start = data["step_start"]
    step_end = data["step_end"]
    guidance_scale = data["guidance_scale"]
    transformer_path = data["transformer_path"]
    expand_timesteps = data["expand_timesteps"]
    mask = data["mask"]

    DTYPE = torch.bfloat16
    prompt_embeds = prompt_embeds.to(DTYPE)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)

    # Load config
    config_file = os.path.join(transformer_path, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    compiled_batch_size = config.get("batch_size", 1)

    print(f"  Compiled batch_size={compiled_batch_size}")

    # Load RoPE
    rope_cache = torch.load(os.path.join(transformer_path, "rope_cache.pt"), weights_only=True)
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(DTYPE)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(DTYPE)

    # Load weights
    print(f"  Loading weights from {transformer_path}...")
    tp_checkpoints = load_sharded_weights(transformer_path, tp_degree)
    cp_checkpoints = prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree)
    del tp_checkpoints
    gc.collect()

    # Load NxDModel
    nxd_model_path = os.path.join(transformer_path, "nxd_model.pt")
    print(f"  Loading NxDModel (TP={tp_degree}, CP={cp_degree}, world_size={world_size})...")
    t0 = time.time()
    nxd_model = NxDModel.load(nxd_model_path, start_rank=0, local_ranks_size=world_size)
    nxd_model.set_weights(cp_checkpoints)
    nxd_model.to_neuron()
    load_time = time.time() - t0
    print(f"  NxDModel loaded in {load_time:.1f}s")

    del cp_checkpoints
    gc.collect()

    # Scheduler
    from diffusers import UniPCMultistepScheduler
    scheduler_config = data["scheduler_config"]
    scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(len(timesteps), device=torch.device("cpu"))
    scheduler.timesteps = timesteps
    if "scheduler_order_list" in data:
        scheduler.order_list = data["scheduler_order_list"]
    if "scheduler_model_outputs" in data:
        scheduler.model_outputs = data["scheduler_model_outputs"]
    if "scheduler_timestep_list" in data:
        scheduler.timestep_list = data["scheduler_timestep_list"]
    if "scheduler_lower_order_nums" in data:
        scheduler.lower_order_nums = data["scheduler_lower_order_nums"]
    if "scheduler_sample" in data:
        scheduler.sample = data["scheduler_sample"]

    # Run
    num_steps = step_end - step_start
    print(f"  Running {num_steps} denoising steps ({step_start}-{step_end-1}), batch_size={compiled_batch_size}...")
    latents, phase_time = run_phase_steps(
        nxd_model, latents, prompt_embeds, negative_prompt_embeds,
        timesteps, step_start, step_end, guidance_scale,
        expand_timesteps, mask, scheduler, rotary_emb_cos, rotary_emb_sin,
        compiled_batch_size,
    )
    print(f"  Phase done. {num_steps} steps in {phase_time:.1f}s")

    output_data = {
        "latents": latents,
        "scheduler_model_outputs": scheduler.model_outputs,
        "scheduler_timestep_list": scheduler.timestep_list,
        "scheduler_lower_order_nums": scheduler.lower_order_nums,
        "scheduler_sample": scheduler.sample,
        "load_time": load_time,
        "phase_time": phase_time,
    }
    torch.save(output_data, output_path)

    total_time = time.time() - t_total
    print(f"  Worker total time: {total_time:.1f}s (load: {load_time:.1f}s, denoise: {phase_time:.1f}s)")


def main_combined(data):
    """Combined MoE mode: load T1, run, replace_weights T2, run, exit."""
    t_total = time.time()

    latents = data["latents"]
    prompt_embeds = data["prompt_embeds"]
    negative_prompt_embeds = data["negative_prompt_embeds"]
    timesteps = data["timesteps"]
    switch_idx = data["switch_idx"]
    guidance_scale_high = data["guidance_scale_high"]
    guidance_scale_low = data["guidance_scale_low"]
    transformer_1_path = data["transformer_1_path"]
    transformer_2_path = data["transformer_2_path"]
    expand_timesteps = data["expand_timesteps"]
    mask = data["mask"]

    DTYPE = torch.bfloat16
    prompt_embeds = prompt_embeds.to(DTYPE)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)

    # Load config from transformer_1
    config_file = os.path.join(transformer_1_path, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    compiled_batch_size = config.get("batch_size", 1)

    print(f"  Combined MoE mode: batch_size={compiled_batch_size}")

    # Load RoPE (same for both transformers)
    rope_cache = torch.load(os.path.join(transformer_1_path, "rope_cache.pt"), weights_only=True)
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(DTYPE)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(DTYPE)

    # Pre-load transformer_2 weights (while we have memory)
    print(f"  Pre-loading transformer_2 weights from {transformer_2_path}...")
    t2_tp_checkpoints = load_sharded_weights(transformer_2_path, tp_degree)
    t2_cp_checkpoints = prepare_cp_checkpoints(t2_tp_checkpoints, tp_degree, cp_degree)
    del t2_tp_checkpoints
    gc.collect()

    # Load transformer_1
    print(f"  Loading transformer_1 weights from {transformer_1_path}...")
    t1_tp_checkpoints = load_sharded_weights(transformer_1_path, tp_degree)
    t1_cp_checkpoints = prepare_cp_checkpoints(t1_tp_checkpoints, tp_degree, cp_degree)
    del t1_tp_checkpoints
    gc.collect()

    nxd_model_path = os.path.join(transformer_1_path, "nxd_model.pt")
    print(f"  Loading NxDModel (TP={tp_degree}, CP={cp_degree}, world_size={world_size})...")
    t0 = time.time()
    nxd_model = NxDModel.load(nxd_model_path, start_rank=0, local_ranks_size=world_size)
    nxd_model.set_weights(t1_cp_checkpoints)
    nxd_model.to_neuron()
    load_time = time.time() - t0
    print(f"  NxDModel loaded with transformer_1 in {load_time:.1f}s")

    del t1_cp_checkpoints
    gc.collect()

    # Setup scheduler
    from diffusers import UniPCMultistepScheduler
    scheduler_config = data["scheduler_config"]
    scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(len(timesteps), device=torch.device("cpu"))
    scheduler.timesteps = timesteps
    if "scheduler_order_list" in data:
        scheduler.order_list = data["scheduler_order_list"]
    if "scheduler_model_outputs" in data:
        scheduler.model_outputs = data["scheduler_model_outputs"]
    if "scheduler_timestep_list" in data:
        scheduler.timestep_list = data["scheduler_timestep_list"]
    if "scheduler_lower_order_nums" in data:
        scheduler.lower_order_nums = data["scheduler_lower_order_nums"]
    if "scheduler_sample" in data:
        scheduler.sample = data["scheduler_sample"]

    # ---- Phase 1: High-noise steps with transformer_1 ----
    print(f"\n  Running {switch_idx} high-noise steps (transformer_1)...")
    latents, phase1_time = run_phase_steps(
        nxd_model, latents, prompt_embeds, negative_prompt_embeds,
        timesteps, 0, switch_idx, guidance_scale_high,
        expand_timesteps, mask, scheduler, rotary_emb_cos, rotary_emb_sin,
        compiled_batch_size,
    )
    print(f"  Phase 1 done: {switch_idx} steps in {phase1_time:.1f}s")

    # ---- Swap weights to transformer_2 ----
    remaining_steps = len(timesteps) - switch_idx
    swap_time = 0.0
    phase2_time = 0.0
    if remaining_steps > 0:
        print(f"\n  Swapping weights to transformer_2 via replace_weights()...")
        t_swap = time.time()
        nxd_model.replace_weights(t2_cp_checkpoints)
        swap_time = time.time() - t_swap
        print(f"  Weight swap completed in {swap_time:.1f}s")

        del t2_cp_checkpoints
        gc.collect()

        # ---- Phase 2: Low-noise steps with transformer_2 ----
        print(f"\n  Running {remaining_steps} low-noise steps (transformer_2)...")
        latents, phase2_time = run_phase_steps(
            nxd_model, latents, prompt_embeds, negative_prompt_embeds,
            timesteps, switch_idx, len(timesteps), guidance_scale_low,
            expand_timesteps, mask, scheduler, rotary_emb_cos, rotary_emb_sin,
            compiled_batch_size,
        )
        print(f"  Phase 2 done: {remaining_steps} steps in {phase2_time:.1f}s")

    total_phase_time = phase1_time + swap_time + phase2_time

    output_data = {
        "latents": latents,
        "scheduler_model_outputs": scheduler.model_outputs,
        "scheduler_timestep_list": scheduler.timestep_list,
        "scheduler_lower_order_nums": scheduler.lower_order_nums,
        "scheduler_sample": scheduler.sample,
        "load_time": load_time,
        "swap_time": swap_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase_time": total_phase_time,
    }
    torch.save(output_data, output_path)

    total_time = time.time() - t_total
    print(f"\n  Worker total: {total_time:.1f}s (load: {load_time:.1f}s, swap: {swap_time:.1f}s, "
          f"phase1: {phase1_time:.1f}s, phase2: {phase2_time:.1f}s)")


if __name__ == "__main__":
    main()
