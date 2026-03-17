"""
Long-running subprocess worker for denoising on Neuron.

Loads model ONCE at startup, then processes multiple inference requests
via stdin/stdout protocol. Eliminates model load/swap overhead between inferences.

Protocol:
  Startup: loads model, prints "READY" to stdout
  Request: reads "denoise <input.pt> <output.pt>" from stdin
           runs denoising steps, saves result, prints "DONE" to stdout
  Shutdown: reads "exit" from stdin, exits

Usage (called programmatically by run_wan2.2_t2v_a14b_persistent.py):
    NEURON_RT_VISIBLE_CORES=8-15 python -m neuron_wan2_2_t2v_a14b.persistent_denoise_worker \
        --transformer_path /path/to/transformer --env_config /path/to/env.json
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_path", required=True)
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--warmup", action="store_true",
                        help="Run a dummy forward pass before READY to establish NCCL ring")
    args = parser.parse_args()

    # Set env vars before Neuron imports
    if args.env_config:
        with open(args.env_config) as f:
            env_config = json.load(f)
        for k, v in env_config.items():
            os.environ[k] = str(v)

    import gc
    import torch
    import torch_neuronx
    from neuronx_distributed import NxDModel
    from safetensors.torch import load_file

    transformer_path = args.transformer_path

    # Load config
    with open(os.path.join(transformer_path, "config.json")) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    compiled_batch_size = config.get("batch_size", 1)

    print(f"[Worker] Loading transformer from {transformer_path}", flush=True)
    print(f"[Worker] TP={tp_degree}, CP={cp_degree}, ws={world_size}, batch={compiled_batch_size}", flush=True)

    # Load RoPE
    rope_cache = torch.load(os.path.join(transformer_path, "rope_cache.pt"), weights_only=True)
    DTYPE = torch.bfloat16
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(DTYPE)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(DTYPE)

    # Load weights
    def load_sharded_weights(model_path, tp_deg):
        weights_path = os.path.join(model_path, "weights")
        sharded = []
        for rank in range(tp_deg):
            ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            raw = load_file(ckpt_path)
            sharded.append({k: v for k, v in raw.items() if 'master_weight' not in k})
        return sharded

    def prepare_cp_checkpoints(tp_ckpts, tp_deg, cp_deg):
        out = []
        for cp_rank in range(cp_deg):
            for tp_rank in range(tp_deg):
                world_rank = cp_rank * tp_deg + tp_rank
                ckpt = {k: v.clone() for k, v in tp_ckpts[tp_rank].items()}
                ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
                out.append(ckpt)
        return out

    tp_ckpts = load_sharded_weights(transformer_path, tp_degree)
    cp_ckpts = prepare_cp_checkpoints(tp_ckpts, tp_degree, cp_degree)
    del tp_ckpts
    gc.collect()

    # Load NxDModel
    t0 = time.time()
    nxd_model = NxDModel.load(
        os.path.join(transformer_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=world_size,
    )
    nxd_model.set_weights(cp_ckpts)
    nxd_model.to_neuron()
    load_time = time.time() - t0
    print(f"[Worker] Model loaded in {load_time:.1f}s", flush=True)

    del cp_ckpts
    gc.collect()

    # Helper: run one transformer forward pass
    def run_step(hidden_states, timestep, encoder_hidden_states):
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

    # Optional warmup forward pass to establish NCCL ring
    if args.warmup:
        print("[Worker] Running warmup forward pass to init NCCL...", flush=True)
        t0 = time.time()
        # Build dummy inputs from config
        in_channels = config.get("in_channels", 16)
        latent_frames = config.get("latent_frames", 21)
        latent_h = config.get("latent_height", 60)
        latent_w = config.get("latent_width", 104)
        seq_len = config.get("seq_len", latent_frames * latent_h * latent_w)
        max_seq = config.get("max_sequence_length", 512)
        # encoder_dim is T5 output dim (4096), NOT transformer hidden_size (5120)
        encoder_dim = config.get("encoder_dim", 4096)
        dummy_hs = torch.zeros(1, in_channels, latent_frames, latent_h, latent_w, dtype=DTYPE)
        dummy_ts = torch.zeros(1, dtype=torch.float32)
        dummy_enc = torch.zeros(1, max_seq, encoder_dim, dtype=DTYPE)
        try:
            _ = nxd_model(dummy_hs, dummy_ts, dummy_enc,
                          rotary_emb_cos, rotary_emb_sin)
            print(f"[Worker] Warmup done in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"[Worker] Warmup FAILED: {type(e).__name__}: {e}", flush=True)
            print(f"[Worker] Shapes: hs={list(dummy_hs.shape)}, ts={list(dummy_ts.shape)}, enc={list(dummy_enc.shape)}", flush=True)
            traceback.print_exc()
            # Don't exit - continue to READY, warmup is best-effort

    # Signal ready
    print("READY", flush=True)

    # Command loop
    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        if parts[0] == "exit":
            print("[Worker] Exiting.", flush=True)
            break

        if parts[0] == "denoise" and len(parts) == 3:
            input_path, output_path = parts[1], parts[2]
            try:
                data = torch.load(input_path, weights_only=False)
                latents = data["latents"]
                prompt_embeds = data["prompt_embeds"].to(DTYPE)
                negative_prompt_embeds = data["negative_prompt_embeds"]
                if negative_prompt_embeds is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)
                timesteps = data["timesteps"]
                step_start = data["step_start"]
                step_end = data["step_end"]
                guidance_scale = data["guidance_scale"]
                expand_timesteps = data["expand_timesteps"]
                mask = data["mask"]

                # Restore scheduler
                from diffusers import UniPCMultistepScheduler
                scheduler = UniPCMultistepScheduler.from_config(data["scheduler_config"])
                scheduler.set_timesteps(len(timesteps), device=torch.device("cpu"))
                scheduler.timesteps = timesteps
                for key in ["scheduler_order_list", "scheduler_model_outputs",
                            "scheduler_timestep_list", "scheduler_lower_order_nums",
                            "scheduler_sample"]:
                    if key in data:
                        attr = key.replace("scheduler_", "")
                        setattr(scheduler, attr, data[key])

                # Run denoising steps
                num_steps = step_end - step_start
                print(f"[Worker] Running {num_steps} steps ({step_start}-{step_end-1}), gs={guidance_scale}", flush=True)
                t0 = time.time()

                for i in range(step_start, step_end):
                    t = timesteps[i]
                    latent_input = latents.to(DTYPE)

                    if expand_timesteps:
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                        ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        ts = t.expand(latents.shape[0])

                    noise_pred = run_step(latent_input, ts, prompt_embeds)
                    noise_uncond = run_step(latent_input, ts, negative_prompt_embeds)

                    noise_pred = noise_uncond.float() + guidance_scale * (noise_pred.float() - noise_uncond.float())
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    step_num = i - step_start + 1
                    if step_num % 5 == 0 or i == step_end - 1 or step_num == 1:
                        elapsed = time.time() - t0
                        print(f"[Worker]   Step {step_num}/{num_steps} (t={t.item():.0f}) - {elapsed:.1f}s", flush=True)

                phase_time = time.time() - t0
                print(f"[Worker] Done: {num_steps} steps in {phase_time:.1f}s", flush=True)

                # Save output
                torch.save({
                    "latents": latents,
                    "scheduler_model_outputs": scheduler.model_outputs,
                    "scheduler_timestep_list": scheduler.timestep_list,
                    "scheduler_lower_order_nums": scheduler.lower_order_nums,
                    "scheduler_sample": scheduler.sample,
                    "phase_time": phase_time,
                }, output_path)

                print("DONE", flush=True)

            except Exception as e:
                print(f"[Worker] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("ERROR", flush=True)
        else:
            print(f"[Worker] Unknown command: {cmd}", flush=True)


if __name__ == "__main__":
    main()
