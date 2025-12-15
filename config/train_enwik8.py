"""
Configuration for training the Transformer-XL model on the enwik8 dataset.

Each configuration is a Python file that defines a TrainConfig object.
It overrides the default configuration with the values specified in the file.
"""

import torch
import platform

from src.config import TrainConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
compile = device == "cuda" and platform.system() == "Linux"

config = TrainConfig(
    dataset="enwik8",
    out_dir="out-enwik8",
    wandb_log=True,
    wandb_project="enwik8",
    wandb_run_name="enwik8-baseline",
    batch_size=12,
    block_size=1024,
    gradient_accumulation_steps=8,
    max_iters=100_000,
    lr_decay_iters=100_000,
    eval_interval=1000,
    eval_iters=200,
    log_interval=10,
    weight_decay=1e-1,
    device=device,
    compile=compile,
)
