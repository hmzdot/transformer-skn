import torch
import platform

dataset = "enwik8"
out_dir = "out-enwik8"

wandb_log = False
wandb_project = "enwik8"
wandb_run_name = "enwik8-baseline"

# 12 batch size * 1024 block size * 8 gradaccum = 98,304
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 8

# total number of tokens is 100M
# 100M tokens / 98,304 tokens/iter = 1017.28 iters
max_iters = 1017
lr_decay_iters = 1017

# eval stuff
eval_interval = 1000
eval_iters = 1  # originally 200
log_interval = 10

# weight decay
weight_decay = 1e-1


device = "cuda" if torch.cuda.is_available() else "cpu"

print(platform.system())
is_linux = platform.system() == "Linux"
compile = device == "cuda" and is_linux
