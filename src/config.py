from dataclasses import dataclass


@dataclass
class TrainConfig:
    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = "owt"
    wandb_run_name = "txl"
    # data
    dataset = "enwik8"
    gradient_accumulation_steps = 5 * 8
    batch_size = 12
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    bias = False
    # adamw optimizer
    learning_rate = 6e-4
    max_iters = 600000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = 600000
    min_lr = 6e-5
    backend = "nccl"
    device = "cuda"
    dtype = "float16"
    compile = True


@dataclass
class SampleConfig:
    out_dir = "out"
    start = "\n"
    num_samples = 10
    max_new_tokens = 500
    temperature = 0.8
    top_k = 200
    seed = 1337
    device = "cuda"
    dtype = "float16"
    compile = False
