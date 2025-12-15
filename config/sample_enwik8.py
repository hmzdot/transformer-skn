"""
Configuration for sampling from the trained Transformer-XL model on the enwik8 dataset.

Each configuration is a Python file that defines a TrainConfig object.
It overrides the default configuration with the values specified in the file.
"""

import torch
from src.config import SampleConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

config = SampleConfig(
    out_dir="out-enwik8",
    start="\n",
    num_samples=10,
    max_new_tokens=500,
    temperature=0.8,
    top_k=200,
    seed=1337,
    device="cuda",
    dtype="float16",
    compile=False,
)
