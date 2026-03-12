"""
Subprocess worker for running a single denoising phase on Neuron.

Supports both batch_size=1 (two forward passes per step) and batch_size=2
(single batched forward for CFG). batch_size is auto-detected from the
compiled model's config.json.

Usage (called programmatically, not directly):
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
        # Expand timestep for batch=2
        timestep = timestep.expand(2)

    # Batch hidden_states: [1,...] -> [2,...]
    hidden_states_batched = hidden_states.expand(2, *hidden_states.shape[1:]).contiguous()
    # Batch encoder_hidden_states: stack prompt + negative
    encoder_hidden_states_batched = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    # Expand RoPE for batch=2
    rotary_emb_cos_batched = rotary_emb_cos.expand(2, *rotary_emb_cos.shape[1:]).contiguous()
    rotary_emb_sin_batched = rotary_emb_sin.expand(2, *rotary_emb_sin.shape[1:]).contiguous()

    output = nxd_model(
        hidden_states_batched, timestep, encoder_hidden_states_batched,
        rotary_emb_cos_batched, rotary_emb_sin_batched,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]

    # Split: output[0] = prompt (noise_pred), output[1] = negative (noise_uncond)
    noise_pred = output[0:1]
    noise_uncond = output[1:2]
    return noise_pred, noise_uncond


def main():
    t_total = time.time()

    # Load phase input data
    data = torch.load(input_path, weights_only=False)
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

    # Import scheduler for step() calls
    from diffusers import UniPCMultistepScheduler
    scheduler_config = data["scheduler_config"]
    scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(len(timesteps), device=torch.device("cpu"))
    # Restore scheduler state
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

    # Run denoising steps
    num_steps = step_end - step_start
    print(f"  Running {num_steps} denoising steps ({step_start}-{step_end-1}), batch_size={compiled_batch_size}...")
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
            # Batched CFG: single forward pass with batch=2
            noise_pred, noise_uncond = run_transformer_step_batched(
                nxd_model, latent_input, ts,
                prompt_embeds, negative_prompt_embeds,
                rotary_emb_cos, rotary_emb_sin,
            )
        else:
            # Original: two separate forward passes
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
    print(f"  Phase done. {num_steps} steps in {phase_time:.1f}s")

    # Save outputs
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


if __name__ == "__main__":
    main()
