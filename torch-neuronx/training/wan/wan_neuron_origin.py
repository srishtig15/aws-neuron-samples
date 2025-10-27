#!/usr/bin/env python
# coding=utf-8

################################################################################
###                                                                          ###
###                                 IMPORTS                                  ###
###                                                                          ###
################################################################################

# System
import gc
import os
import sys
import pathlib
import random
from glob import glob
from typing import Union

# Neuron
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams
from neuronx_distributed.parallel_layers import parallel_state
import torch_xla.core.xla_model as xm

# General ML stuff
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# For measuring throughput
import queue
import time

# LR scheduler
from diffusers.optimization import get_scheduler

# Dataset
# For logging and benchmarking
import time

# Command line args
import argparse

# Multicore
import torch.distributed as dist
import torch_xla.distributed.parallel_loader as xpl
import torch_xla.runtime as xr
from torch.utils.data.distributed import DistributedSampler

from diffusers import AutoencoderKLWan, WanPipeline

from torch_xla.amp.syncfree.adamw import AdamW
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

from neuron_parallel_utils import shard_transformer_feedforward, shard_transformer3d_attn

from importlib.metadata import version
is_pt_2_x = version("torch") >= "2.0"

if is_pt_2_x:
    from torch_xla.utils.checkpoint import checkpoint as torch_xla_grad_checkpoint
    # torch.utils.checkpoint.checkpoint doesn't work on PT 2.1
    # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/torch-neuronx/introducing-pytorch-2-1.html?highlight=2.1#attributeerror-module-torch-has-no-attribute-xla-failure
    def _call_grad_checkpoint(*args, **kwargs):
        kwargs["use_reentrant"] = True
        return torch_xla_grad_checkpoint(*args, **kwargs)

    torch.utils.checkpoint.checkpoint = _call_grad_checkpoint



################################################################################
###                                                                          ###
###                           CONSTANTS, ENV SETUP                           ###
###                                                                          ###
################################################################################

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
# Increase parameter wrapping threshold to avoid tuple arguments issue on Neuron
os.environ["XLA_PARAMETER_WRAPPING_THREADSHOLD"] = "10000"

##### Neuron compiler flags #####
# --model-type=transformer: Enable transformer-specific optimizations
# --enable-saturate-infinity: Needed for correctness
# --distribution-strategy=llm-training: Enable large model training optimizations
# -O1: Use optimization level 1 to minimize memory usage during compilation
# --internal-hlo2tensorizer-options: Additional memory optimization options
compiler_flags = """ --target=trn2 --lnc=2 --retry_failed_compilation --cache_dir="./compiler_cache" --model-type=transformer --enable-saturate-infinity """  # --internal-hlo2tensorizer-options='--fuse-dot-logistic=false'

os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

# Path to where this file is located
curr_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(curr_dir)

image_column_name = "image"
caption_column_name = "text"

LOSS_FILE_FSTRING = "LOSSES-RANK-{RANK}.txt"


################################################################################
###                                                                          ###
###                           HELPER FUNCTIONS                               ###
###                                                                          ###
################################################################################

def shard_wan_transformer(transformer, tp_degree):
    """Apply tensor parallelism to Wan transformer model"""
    xm.master_print(f"Sharding Wan transformer with TP degree {tp_degree}")

    # Shard each transformer block
    for block_id, block in enumerate(transformer.blocks):
        xm.master_print(f"  Sharding block {block_id}")

        # Shard attention layers
        if hasattr(block, 'attn1'):
            block.attn1 = shard_transformer3d_attn(tp_degree, block.attn1)
        if hasattr(block, 'attn2'):
            block.attn2 = shard_transformer3d_attn(tp_degree, block.attn2)

        # Shard feedforward layer
        if hasattr(block, 'ff'):
            block.ff = shard_transformer_feedforward(block.ff)

    xm.master_print("Finished sharding Wan transformer")
    return transformer

# For measuring throughput
class Throughput:
    def __init__(self, batch_size=8, data_parallel_degree=2, grad_accum_usteps=1, moving_avg_window_size=10):
        self.inputs_per_training_step = batch_size * data_parallel_degree * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    # Record a training step - to be called anytime we call optimizer.step()
    def step(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        return

    # Returns the throughput measured over the last moving_avg_window_size steps
    def get_throughput(self):
        throughput = self.moving_avg_window.qsize() * self.inputs_per_training_step / self.window_time
        return throughput


# Patch ZeRO Bug - need to explicitly initialize the clip_value as the dtype we want
@torch.no_grad()
def _clip_grad_norm(
    self,
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2.0,
) -> torch.Tensor:
    """
    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = self._calc_grad_norm(norm_type)

    clip_coeff = torch.tensor(
        max_norm, device=self.device) / (
            total_norm + 1e-6)
    clip_value = torch.where(clip_coeff < 1, clip_coeff,
                                torch.tensor(1., dtype=clip_coeff.dtype, device=self.device))
    for param_group in self.base_optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                p.grad.detach().mul_(clip_value)

ZeroRedundancyOptimizer._clip_grad_norm = _clip_grad_norm


# Saves a pipeline to the specified dir using HuggingFace's built-in methods, suitable for loading
# as a pretrained model in an inference script
def save_pipeline(results_dir, pipe):
    xm.master_print(f"Saving trained model to dir {results_dir}")

    if xm.is_master_ordinal():
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Wait for any pending XLA operations to complete before copying
        xm.mark_step()
        xm.wait_device_ops()

        # Get the transformer state dict and convert to CPU
        xm.master_print("Extracting transformer state dict...")
        transformer_state = pipe.transformer.state_dict()

        # Convert XLA tensors to CPU tensors
        xm.master_print("Converting tensors to CPU...")
        cpu_transformer_state = {}
        for k, v in transformer_state.items():
            cpu_transformer_state[k] = xm._maybe_convert_to_cpu(v)

        # Create a temporary copy of the pipeline for saving
        import copy
        xm.master_print("Creating CPU copy of transformer for saving...")

        # Save transformer separately using torch.save (works with XLA)
        transformer_save_path = os.path.join(results_dir, "transformer")
        os.makedirs(transformer_save_path, exist_ok=True)

        xm.master_print("Saving transformer state dict...")
        torch.save(cpu_transformer_state, os.path.join(transformer_save_path, "diffusion_pytorch_model.bin"))

        # Save transformer config
        if hasattr(pipe.transformer, 'config'):
            import json
            config_dict = dict(pipe.transformer.config)
            with open(os.path.join(transformer_save_path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

        # Save other components (VAE, text encoder, etc.)
        xm.master_print("Saving other pipeline components...")

        # Remember original devices
        vae_device = None
        text_encoder_device = None

        # Save VAE - might be on XLA device, move to CPU first
        if pipe.vae is not None:
            try:
                # Get current device
                vae_device = next(pipe.vae.parameters()).device
                vae_cpu = pipe.vae.to('cpu')
                vae_cpu.save_pretrained(os.path.join(results_dir, "vae"))
                del vae_cpu
                # Move back to original device
                if vae_device is not None and str(vae_device) != 'cpu':
                    pipe.vae = pipe.vae.to(vae_device)
            except Exception as e:
                xm.master_print(f"Warning: Could not save VAE: {e}")

        # Save text encoder - might be on XLA device, move to CPU first
        if pipe.text_encoder is not None:
            try:
                # Get current device
                text_encoder_device = next(pipe.text_encoder.parameters()).device
                text_encoder_cpu = pipe.text_encoder.to('cpu')
                text_encoder_cpu.save_pretrained(os.path.join(results_dir, "text_encoder"))
                del text_encoder_cpu
                # Move back to original device
                if text_encoder_device is not None and str(text_encoder_device) != 'cpu':
                    pipe.text_encoder = pipe.text_encoder.to(text_encoder_device)
            except Exception as e:
                xm.master_print(f"Warning: Could not save text_encoder: {e}")

        # Save tokenizer (always on CPU)
        if pipe.tokenizer is not None:
            try:
                pipe.tokenizer.save_pretrained(os.path.join(results_dir, "tokenizer"))
            except Exception as e:
                xm.master_print(f"Warning: Could not save tokenizer: {e}")

        # Save scheduler (always on CPU)
        if pipe.scheduler is not None:
            try:
                pipe.scheduler.save_pretrained(os.path.join(results_dir, "scheduler"))
            except Exception as e:
                xm.master_print(f"Warning: Could not save scheduler: {e}")

        # Save model index
        xm.master_print("Saving model index...")
        model_index = {
            "_class_name": pipe.__class__.__name__,
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5Tokenizer"],
            "scheduler": ["diffusers", "DDPMScheduler"]
        }
        with open(os.path.join(results_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)

        xm.master_print("Pipeline saved successfully")

    xm.master_print(f"Done saving trained model to dir {results_dir}")
    return


# Saves a checkpoint of the unet and optimizer to the directory specified
# If ZeRO-1 optimizer sharding is enabled, each ordinal needs to save its own checkpoint of the optimizer
def save_checkpoint(results_dir, unet, optimizer, epoch, step, cumulative_step):
    # Save UNet state - only the master needs to save as UNet state is identical between workers
    if xm.is_master_ordinal():
        checkpoint_path = os.path.join(results_dir, f"checkpoint-unet-epoch_{epoch}-step_{step}-cumulative_train_step_{cumulative_step}.pt")
        xm.master_print(f"Saving UNet state checkpoint to {checkpoint_path}")
        data = {
            'epoch': epoch,
            'step': step,
            'cumulative_train_step': cumulative_step,
            'unet_state_dict': unet.state_dict(),
        }
        # Copied from https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/dp_bert_hf_pretrain/dp_bert_large_hf_pretrain_hdf5.py
        # Not sure if it's strictly needed
        cpu_data = xm._maybe_convert_to_cpu(data)
        torch.save(cpu_data, checkpoint_path)
        del(cpu_data)
        xm.master_print(f"Done saving UNet state checkpoint to {checkpoint_path}")

    # Save optimizer state
    # Under ZeRO optimizer sharding each worker needs to save the optimizer state
    # as each has its own unique state
    checkpoint_path = os.path.join(results_dir, f"checkpoint-optimizer-epoch_{epoch}-step_{step}-cumulative_train_step_{cumulative_step}-rank_{xr.global_ordinal()}.pt")
    xm.master_print(f"Saving optimizer state checkpoint to {checkpoint_path} (other ranks will ahve each saved their own state checkpoint)")
    data = {
        'epoch': epoch,
        'step': step,
        'cumulative_train_step': cumulative_step,
        'optimizer_state_dict': optimizer.state_dict()
    }
    cpu_data = data
    # Intentionally don't move the data to CPU here - it causes XLA to crash
    # later when loading the optimizer checkpoint once the optimizer gets run
    torch.save(cpu_data, checkpoint_path)
    del(cpu_data)
    xm.master_print(f"Done saving optimizer state checkpoint to {checkpoint_path}")

    # Make the GC collect the CPU data we deleted so the memory actually gets freed
    gc.collect()
    xm.master_print("Done saving checkpoints!")


# Loads a checkpoint of the unet and optimizer from the directory specified
# If ZeRO-1 optimizer sharding is enabled, each ordinal needs to load its own checkpoint of the optimizer
# Returns a tuple of (epoch, step, cumulative_train_step)
def load_checkpoint(results_dir, unet, optimizer, device, resume_step):
    # Put an asterisk in for globbing if the user didn't specify a resume_step
    if resume_step is None:
        resume_step = "*"
    unet_checkpoint_filenames = glob(os.path.join(results_dir, f"checkpoint-unet-epoch_*-step_*-cumulative_train_step_{resume_step}.pt"))
    optimizer_checkpoint_filenames = glob(os.path.join(results_dir, f"checkpoint-optimizer-epoch_*-step_*-cumulative_train_step_{resume_step}-rank_{xr.global_ordinal()}.pt"))

    unet_checkpoint_filenames.sort()
    optimizer_checkpoint_filenames.sort()

    # Load UNet checkpoint
    checkpoint_path = unet_checkpoint_filenames[-1]
    xm.master_print(f"Loading UNet checkpoint from path {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
    ret = (checkpoint['epoch'], checkpoint['step'], checkpoint['cumulative_train_step'])
    del(checkpoint)

    # Load optimizer checkpoint
    checkpoint_path = optimizer_checkpoint_filenames[-1]
    xm.master_print(f"Loading optimizer checkpoint from path {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(optimizer, torch.nn.Module):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=True)
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    assert checkpoint['epoch'] == ret[0] and checkpoint['step'] == ret[1] and checkpoint['cumulative_train_step'] == ret[2], \
        "UNet checkpoint and optimizer checkpoint do not agree on the epoch, step, or cumulative_train_step!"
    del(checkpoint)

    gc.collect()

    xm.master_print("Done loading checkpoints!")
    return ret


# Seed various RNG sources that need to be seeded to make training deterministic
# WARNING: calls xm.rendezvous() internally
def seed_rng(device):
    LOCAL_RANK = xr.global_ordinal()
    xm.rendezvous('start-seeding-cpu')
    torch.manual_seed(9999 + LOCAL_RANK)
    random.seed(9999+ LOCAL_RANK)
    np.random.seed(9999 + LOCAL_RANK)

    xm.rendezvous('start-seeding-device')
    xm.set_rng_state(9999 + LOCAL_RANK, device=device)
    # TODO: not sure if we need to print the RNG state on CPU to force seeding to actually happen
    xm.master_print(f"xla rand state after setting RNG state {xm.get_rng_state(device=device)}\n")
    xm.rendezvous('seeding-device-done')

    xm.master_print("Done seeding CPU and device RNG!")




################################################################################
###                                                                          ###
###                           MAIN TRAINING FUNCTION                         ###
###                                                                          ###
################################################################################

def forward_preprocess(data, pipe, device, tokenizer, text_encoder, vae, use_gradient_checkpointing=True, max_frames=4):
    """预处理UnifiedDataset的数据为WAN模型需要的格式"""

    # data["video"] is a list of batches, each batch is a list of PIL Images
    # Get the first batch's first image to extract dimensions
    first_batch_videos = data["video"][0] if len(data["video"]) > 0 else []
    if isinstance(first_batch_videos, list) and len(first_batch_videos) > 0:
        # PIL Image objects have .size as a tuple (width, height)
        height = first_batch_videos[0].size[1]
        width = first_batch_videos[0].size[0]
        num_frames = min(len(first_batch_videos), max_frames)  # Limit frames to reduce memory
    else:
        height = 128  # 480
        width = 128  # 832
        num_frames = min(81, max_frames)

    # Process text prompts with tokenizer and text encoder
    batch_size = len(data["prompt"])
    text_inputs = tokenizer(
        data["prompt"],
        padding="max_length",
        max_length=512,  # tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input_ids)[0]

    # Process video frames with VAE
    # For now, we'll prepare the video data in a format similar to what WAN expects
    # This is a simplified version - you may need to adjust based on your specific needs
    video_frames = data["video"]  # List of lists of PIL Images

    # Convert PIL images to tensors and encode with VAE
    # Note: This is a placeholder - you'll need to implement proper video encoding
    # based on how WAN model expects the latents
    # print('encoder_hidden_states:', encoder_hidden_states.shape)  # torch.Size([1, 512, 4096])

    return {
        "video_frames": video_frames,
        "encoder_hidden_states": encoder_hidden_states,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "batch_size": batch_size,
    }

def train(args):
    LOCAL_RANK = xr.global_ordinal()

    # Get max_frames from args or use default
    max_frames = getattr(args, 'max_frames', 2)
    xm.master_print(f'Using max_frames={max_frames} to limit video frames for memory efficiency')

    # Initialize tensor parallelism
    tp_degree = getattr(args, 'tensor_parallel_degree', 4)
    xm.master_print(f'Initializing tensor parallelism with degree {tp_degree}')
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    xm.master_print(f'TP rank: {parallel_state.get_tensor_model_parallel_rank()}, TP size: {parallel_state.get_tensor_model_parallel_size()}')

    # Create all the components of our model pipeline and training loop
    xm.master_print('Building training loop components')

    device = xm.xla_device()

    t = torch.tensor([0.1]).to(device=device)
    xm.mark_step()
    xm.master_print(f"Initialized device, t={t.detach().to(device='cpu')}")

    # Warning: calls xm.rendezvous() internally
    seed_rng(device)

    if not xm.is_master_ordinal(): xm.rendezvous('prepare')

    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # noise_scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    # tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    noise_scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet = pipe.transformer
    # if os.getenv('NEURON_RT_STOCHASTIC_ROUNDING_EN', None):
    #     text_encoder = text_encoder.to(torch.bfloat16)
    #     # vae = vae.to(torch.bfloat16)
    #     unet = unet.to(torch.bfloat16)
    unet.requires_grad_(True)

    xm.master_print("Enabling gradient checkpointing")
    unet.enable_gradient_checkpointing()

    # Apply tensor parallelism sharding to the transformer
    xm.master_print("Applying tensor parallelism to transformer")
    unet = shard_wan_transformer(unet, tp_degree)

    optim_params = unet.parameters()

    # IMPORTANT: need to move unet to device before we create the optimizer for the optimizer to be training the right parameters (on-device)
    unet.train()
    unet.to(device)

    # Setup VAE and text encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    # Keep VAE on CPU - it doesn't need to run on Trainium
    vae.requires_grad_(False)
    vae.eval()
    # Keep VAE on CPU with float32 precision
    vae = vae.to(device='cpu', dtype=torch.float32)
    xm.master_print("VAE kept on CPU for encoding")

    # TODO: parametrize optimizer parameters
    if is_pt_2_x:
        optimizer_class = AdamW_FP32OptimParams
    else:
        optimizer_class = AdamW
    # Use bfloat16 for optimizer to save memory
    optimizer = ZeroRedundancyOptimizer(optim_params, optimizer_class, pin_layout=False, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08, capturable=True, optimizer_dtype=torch.bfloat16)

    # Download the dataset
    xm.master_print('Loading dataset')
    from diffsynth.trainers.unified_dataset import UnifiedDataset

    # Set dataset parameters
    args.dataset_base_path = getattr(args, 'dataset_base_path', 'data/example_video_dataset')
    args.dataset_metadata_path = getattr(args, 'dataset_metadata_path', 'data/example_video_dataset/metadata.csv')
    args.max_pixels = getattr(args, 'max_pixels', 1280*720)
    # Use command-line args if provided, otherwise use defaults
    # Start with 128x128 for compilation, can increase to 256x256, 512x512, etc. for actual training
    args.height = args.train_height if args.train_height is not None else 128
    args.width = args.train_width if args.train_width is not None else 128
    args.dataset_repeat = getattr(args, 'dataset_repeat', 100)
    # Use max_frames for dataset to avoid loading unnecessary frames
    args.num_frames = max_frames
    
    print('args:', args)

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=("video",),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    # Done anything that might trigger a download
    xm.master_print("Executing `if xm.is_master_ordinal(): xm.rendezvous('prepare')`")
    if xm.is_master_ordinal(): xm.rendezvous('prepare')

    def training_metrics_closure(epoch, global_step, loss):
        loss_val = loss.detach().to('cpu').item()
        loss_f.write(f"{LOCAL_RANK} {epoch} {global_step} {loss_val}\n")
        loss_f.flush()

    xm.rendezvous('prepare-to-load-checkpoint')

    loss_filename = f"LOSSES-RANK-{LOCAL_RANK}.txt"

    if args.resume_from_checkpoint:
        start_epoch, start_step, cumulative_train_step = load_checkpoint(args.results_dir, unet, optimizer, device, args.resume_checkpoint_step)
        loss_f = open(loss_filename, 'a')
    else:
        start_epoch = 0
        start_step = 0
        cumulative_train_step = 0

        loss_f = open(loss_filename, 'w')
        loss_f.write("RANK EPOCH STEP LOSS\n")

    xm.rendezvous('done-loading-checkpoint')

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer
    )

    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    parameters = sum([np.prod(p.size()) * p.element_size() for p in parameters]) / (1024 ** 2)
    xm.master_print('Trainable Parameters: %.3fMB' % parameters)
    total_parameters = 0
    components = [text_encoder, vae, unet]
    for component in components:
        total_parameters += sum([np.prod(p.size()) * p.element_size() for p in component.parameters()]) / (1024 ** 2)
    xm.master_print('Total parameters: %.3fMB' % total_parameters)

    train_dataset = dataset
    args.dataset_size = len(train_dataset)

    # Define collate function for UnifiedDataset
    def collate_fn(examples):
        """Process batch data from UnifiedDataset"""
        batch = {
            "video": [ex["video"] for ex in examples],
            "prompt": [ex["prompt"] for ex in examples]
        }
        # Handle optional fields
        for key in ["input_image", "end_image", "reference_image", "vace_reference_image"]:
            if key in examples[0]:
                batch[key] = [ex[key] for ex in examples]
        return batch

    # Create dataloaders
    world_size = xr.world_size()
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=xr.global_ordinal(),
                                           shuffle=True)

    # drop_last=True needed to avoid cases of an incomplete final batch, which would result in new graphs being cut and compiled
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False if train_sampler else True, collate_fn=collate_fn, batch_size=args.batch_size, sampler=train_sampler, drop_last=True
    )

    train_device_loader = xpl.MpDeviceLoader(train_dataloader, device, device_prefetch_size=2)

    xm.master_print('Entering training loop')
    xm.rendezvous('training-loop-start')

    if not is_pt_2_x:
        found_inf = torch.tensor(0, dtype=torch.double, device=device)
    checkpoints_saved = 0

    # Use a moving average window size of 100 so we have a large sample at
    # the end of training
    throughput_helper = Throughput(args.batch_size, world_size, args.gradient_accumulation_steps, moving_avg_window_size=100)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.perf_counter_ns()
        before_batch_load_time = time.perf_counter_ns()
        xm.master_print("####################################")
        xm.master_print(f"###### Starting epoch {epoch} ######")
        xm.master_print("####################################")
        # Add 1 to the start_step so that we don't repeat the step we saved the checkpoint after
        for step, batch in enumerate(train_device_loader, start=(start_step + 1 if epoch == start_epoch else 0)):
            after_batch_load_time = time.perf_counter_ns()

            xm.master_print(f"*** Running epoch {epoch} step {step} (cumulative step {cumulative_train_step})", flush=True)
            start_time = time.perf_counter_ns()

            # Use forward_preprocess to prepare data for WAN model
            # Limit frames to reduce memory usage
            inputs = forward_preprocess(batch, pipe, device, tokenizer, text_encoder, vae, use_gradient_checkpointing=True, max_frames=max_frames)
            
            print('inputs:', inputs)

            # Since WanPipeline doesn't have training_loss, we need to implement the forward pass manually
            # This is a simplified training loop - you'll need to adjust based on WAN's actual training requirements

            # Process video frames to latents
            video_frames = inputs["video_frames"]
            encoder_hidden_states = inputs["encoder_hidden_states"]
            batch_size = inputs["batch_size"]

            # Convert PIL images to tensors for VAE encoding
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            # Process all video frames for each batch
            batch_video_tensors = []
            # Use the max_frames from args (already limited at dataset level)
            for batch_idx, batch_videos in enumerate(video_frames):
                # Convert all frames in this batch to tensors
                frame_tensors = []
                # Limit the number of frames to process
                frames_to_process = batch_videos[:max_frames]
                # print(f"Batch {batch_idx}: Processing {len(frames_to_process)} frames (out of {len(batch_videos)} total)")
                for frame in frames_to_process:
                    frame_tensor = transform(frame)  # [C, H, W]
                    frame_tensors.append(frame_tensor)

                # Stack frames along temporal dimension: [T, C, H, W]
                video_tensor = torch.stack(frame_tensors, dim=0)  # [num_frames, C, H, W]
                # print(f"  Video tensor shape: {video_tensor.shape}")
                batch_video_tensors.append(video_tensor)

            # Stack all batches: [B, T, C, H, W] then permute to [B, C, T, H, W]
            video_batch = torch.stack(batch_video_tensors, dim=0)  # [B, T, C, H, W]
            # print(f"Before permute: {video_batch.shape}")
            # Keep video_batch on CPU for VAE encoding
            video_batch = video_batch.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            # print(f"After permute (B, C, T, H, W): {video_batch.shape}")

            # WAN VAE uses a special caching mechanism
            # It processes video frame by frame with temporal caching
            with torch.no_grad():
                B, C, T, H, W = video_batch.shape
                # print(f"Encoding video with shape (B={B}, C={C}, T={T}, H={H}, W={W})")

                # Initialize VAE cache on CPU - VAE runs on CPU
                if hasattr(vae, 'init_cache'):
                    # print("Initializing VAE cache")
                    vae.init_cache(height=H, width=W, device='cpu')

                # WAN VAE needs special handling
                # Process video in smaller chunks to reduce memory usage

                # First, ensure we have at least 2 frames for the cache mechanism
                CACHE_T = 2
                if T == 1:
                    # If only 1 frame, duplicate it
                    # print(f"Only 1 frame, duplicating to {CACHE_T} frames for cache")
                    video_batch = video_batch.repeat(1, 1, CACHE_T, 1, 1)
                    T = CACHE_T

                # Process frames in smaller chunks to reduce memory
                # Use chunk_size of 1 to minimize memory usage and avoid SB allocation errors
                chunk_size = 1  # Process one frame at a time to minimize SB memory pressure
                latents_list = []

                for i in range(0, T, chunk_size):
                    end_idx = min(i + chunk_size, T)
                    chunk = video_batch[:, :, i:end_idx, :, :]
                    # print(f"Processing frames {i} to {end_idx}, shape {chunk.shape}")

                    # Ensure chunk has at least CACHE_T frames
                    if chunk.shape[2] < CACHE_T:
                        padding = chunk[:, :, -1:, :, :].repeat(1, 1, CACHE_T - chunk.shape[2], 1, 1)
                        chunk = torch.cat([chunk, padding], dim=2)

                    # Encode chunk
                    with torch.no_grad():
                        latent_chunk = vae.encode(chunk).latent_dist.sample()
                        # Only keep the frames we need (not the padding)
                        if end_idx - i < chunk.shape[2]:
                            latent_chunk = latent_chunk[:, :, :end_idx-i, :, :]
                        latents_list.append(latent_chunk)

                        # Clear intermediate tensors to save memory
                        del chunk
                        if i > 0:
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            xm.mark_step()  # Force XLA to clear memory

                latents = torch.cat(latents_list, dim=2)
                # print(f"Combined latents shape: {latents.shape}")

                # Clear intermediate list
                del latents_list

                # Get scaling factor from VAE config
                # WAN VAE might have a different scaling factor
                scaling_factor = None

                # Try different ways to get the scaling factor
                if hasattr(vae.config, 'scaling_factor'):
                    scaling_factor = vae.config.scaling_factor
                elif isinstance(vae.config, dict) and 'scaling_factor' in vae.config:
                    scaling_factor = vae.config['scaling_factor']
                elif hasattr(vae, 'config') and hasattr(vae.config, 'get'):
                    # For FrozenDict
                    scaling_factor = vae.config.get('scaling_factor', None)

                if scaling_factor is None:
                    # WAN models often use 1.0 or no scaling
                    # Check if there's a scale_factor attribute directly on the VAE
                    if hasattr(vae, 'scale_factor'):
                        scaling_factor = vae.scale_factor
                    elif hasattr(vae, 'scaling_factor'):
                        scaling_factor = vae.scaling_factor
                    else:
                        # For WAN video models, the scaling factor is often 1.0
                        # Unlike Stable Diffusion which uses 0.18215
                        scaling_factor = 1.0
                        # print(f"Warning: scaling_factor not found in VAE config, using {scaling_factor} for WAN VAE")

                # print(f"Using VAE scaling factor: {scaling_factor}")
                latents = latents * scaling_factor

            # Move latents from CPU to Trainium device
            latents = latents.to(device)
            # print(f"Moved latents to device: {latents.device}")

            # Ensure latents are in the correct dtype for the model
            # WAN transformer expects bfloat16
            if os.getenv('NEURON_RT_STOCHASTIC_ROUNDING_EN', None):
                latents = latents.to(torch.bfloat16)

            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device)
            timesteps = timesteps.long()  # Ensure timesteps are long type

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Ensure all inputs are in the correct dtype
            noisy_latents = noisy_latents.to(unet.dtype)
            encoder_hidden_states = encoder_hidden_states.to(unet.dtype)

            # Clear original latents to save memory
            del latents

            # Forward pass through transformer/unet with gradient checkpointing
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Clear intermediate tensors
            del model_pred, noisy_latents, noise, encoder_hidden_states

            # Mark step to split the model for better compiler QoR
            xm.mark_step()

            # Backwards pass
            loss.backward()

            xm.mark_step()


            with torch.no_grad():
                # Optimizer
                if (cumulative_train_step + 1) % args.gradient_accumulation_steps == 0:
                    if is_pt_2_x:
                        optimizer.step()
                    else:
                        optimizer.step(found_inf=found_inf)
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    xm.master_print("Finished weight update")
                    throughput_helper.step()
                else:
                    xm.master_print("Accumulating gradients")

            xm.add_step_closure(training_metrics_closure, (epoch, step, loss.detach()), run_async=True)

            xm.mark_step()

            xm.master_print(f"*** Finished epoch {epoch} step {step} (cumulative step {cumulative_train_step})")
            e2e_time = time.perf_counter_ns()
            xm.master_print(f" > E2E for epoch {epoch} step {step} took {e2e_time - before_batch_load_time} ns")

            cumulative_train_step += 1

            # Checkpoint if needed
            if args.checkpointing_steps is not None and cumulative_train_step % args.checkpointing_steps == 0 and cumulative_train_step != 0:
                xm.rendezvous('prepare-to-save-checkpoint')
                save_checkpoint(args.results_dir, unet, optimizer, epoch, step, cumulative_train_step)
                checkpoints_saved += 1
                xm.rendezvous('done-saving-checkpoint')

            before_batch_load_time = time.perf_counter_ns()

            # Only need a handful of training steps for graph extraction. Cut it off so we don't take forever when
            # using a large dataset.
            if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) and cumulative_train_step > 5:
                break

        if args.save_model_epochs is not None and epoch % args.save_model_epochs == 0 and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            save_pipeline(args.results_dir + f"-EPOCH_{epoch}", pipe)

        end_epoch_time = time.perf_counter_ns()
        xm.master_print(f" Entire epoch {epoch} took {(end_epoch_time - start_epoch_time) / (10 ** 9)} s")
        xm.master_print(f" Given {step + 1} many steps, e2e per iteration is {(end_epoch_time - start_epoch_time) / (step + 1) / (10 ** 6)} ms")
        xm.master_print(f"!!! Finished epoch {epoch}")

        # Only need a handful of training steps for graph extraction. Cut it off so we don't take forever when
        # using a large dataset.
        if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) and cumulative_train_step > 5:
            break

    # Save the trained model for use in inference
    xm.rendezvous('finish-training')
    if xm.is_master_ordinal() and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        save_pipeline(os.path.join(args.results_dir, "wan_trained_model_neuron"), pipe)

    loss_f.close()

    xm.master_print(f"!!! Finished all epochs")

    # However, I may need to block here to await the async? How to do that???
    xm.wait_device_ops()

    xm.master_print(f"Average throughput over final 100 training steps was {throughput_helper.get_throughput()} images/s")

    xm.rendezvous('done')
    xm.master_print(f"!!! All done!")

    return




################################################################################
###                                                                          ###
###                             ARG PARSING, MAIN                            ###
###                                                                          ###
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Neuron Wan training script',
                    description='Wan training script for Neuron Trn1/Trn2')
    parser.add_argument('--model', choices=['Wan-AI/Wan2.1-T2V-1.3B-Diffusers', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'], help='Which model to train')
    parser.add_argument('--resolution', choices=[512, 768], type=int, help='Which resolution of model to train')
    parser.add_argument('--batch_size', type=int, help='What per-device microbatch size to use')
    parser.add_argument('--max_frames', type=int, default=2, help='Maximum number of video frames to process (default: 2, use lower values to save memory)')
    parser.add_argument('--tensor_parallel_degree', type=int, default=4, help='Tensor parallelism degree (default: 4 for 64 workers, must divide world_size)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='How many gradient accumulation steps to do (1 for no gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=2000, help='How many epochs to train for')

    # High-resolution training support
    parser.add_argument('--train_height', type=int, default=128, help='Training height (overrides dataset default). Use lower for compilation, then increase gradually.')
    parser.add_argument('--train_width', type=int, default=128, help='Training width (overrides dataset default). Use lower for compilation, then increase gradually.')

    # Arguments for checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=None,
        help=(
            "Save a checkpoint of the training state every X training steps. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--max_num_checkpoints", type=int, default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument("--save_model_epochs", type=int, default=None,
        help=(
            "Save a copy of the trained model every X epochs in a format that can be loaded using HuggingFace's from_pretrained method."
        ))

    # TODO: add ability to specify dir with checkpoints to restore from that is different than the default
    parser.add_argument('--resume_from_checkpoint', action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument('--resume_checkpoint_step', type=int, default=None, help="Which cumulative training step to resume from, looking for checkpoints in the script's work directory. Leave unset to use the latest checkpoint.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    env_world_size = os.environ.get("WORLD_SIZE")

    args = parse_args()

    model_id = args.model
    args.model_id = model_id

    test_name = f"wan_{args.model}_training-{args.resolution}-batch{args.batch_size}-AdamW-{env_world_size}w-zero1_optimizer-grad_checkpointing"

    # Directory to save artifacts to, like checkpoints
    results_dir = os.path.join(curr_dir, test_name + '_results')
    os.makedirs(results_dir, exist_ok=True)
    args.results_dir = results_dir

    dist.init_process_group('xla')
    world_size = xr.world_size()

    args.world_size = world_size

    assert int(world_size) == int(env_world_size), f"Error: world_size {world_size} does not match env_world_size {env_world_size}"

    xm.master_print(f"Starting Wan training script on Neuron, training model {model_id} with the following configuration:")
    for k, v in vars(args).items():
        xm.master_print(f"{k}: {v}")
    xm.master_print(f"World size is {world_size}")
    xm.master_print("")
    xm.master_print(f"## Neuron RT flags ##")
    xm.master_print(f"NEURON_RT_STOCHASTIC_ROUNDING_SEED: {os.getenv('NEURON_RT_STOCHASTIC_ROUNDING_SEED', None)}")
    xm.master_print(f"NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS: {os.getenv('NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS', None)}")
    xm.master_print("")
    xm.master_print(f"## XLA flags ##")
    xm.master_print(f"XLA_IR_DEBUG: {os.getenv('XLA_IR_DEBUG', None)}")
    xm.master_print(f"XLA_HLO_DEBUG: {os.getenv('XLA_HLO_DEBUG', None)}")
    xm.master_print(f"NEURON_RT_STOCHASTIC_ROUNDING_EN: {os.getenv('NEURON_RT_STOCHASTIC_ROUNDING_EN', None)}")

    xm.rendezvous("Entering training function")

    train(args)

    xm.rendezvous("Done training")
