"""
Sample from a trained model
"""

import os
import sys
import pickle
from contextlib import nullcontext
import torch

from .model import TransformerXLConfig, TransformerXL
from .config import SampleConfig

config = SampleConfig()
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    exec(open(config_file).read())

# Setup globals from config, to make nanoGPT implementation work
out_dir = config.out_dir
start = config.start
num_samples = config.num_samples
max_new_tokens = config.max_new_tokens
temperature = config.temperature
top_k = config.top_k
seed = config.seed
device = config.device
dtype = config.dtype
compile = config.compile

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = TransformerXLConfig(**checkpoint["model_args"])
model = TransformerXL(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
assert os.path.exists(meta_path), f"Meta file not found at {meta_path}"

print(f"Loading meta from {meta_path}...")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)


def encode(s: bytes) -> list[int]:
    return list(s)


def decode(l: list[int]) -> bytes:
    return bytes(l)


# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------")
