"""
Prepare the enwik8 dataset for character-level language modeling.
First, downloads the dataset from the URL and unzips it.
Then, it reads the file and converts it to a list of integers.
Finally, it saves the data to train.bin and val.bin.
"""

import os
import pickle
import requests
import zipfile
import numpy as np

from tqdm import tqdm

DATA_URL = "http://mattmahoney.net/dc/enwik8.zip"
VOCAB_SIZE = 256
SPLIT_RATIO = 0.9

file_dir = os.path.dirname(__file__)
input_file_path = os.path.join(file_dir, "enwik8.bin")

# Download and unzip the dataset if not found
if not os.path.exists(input_file_path):
    if not os.path.exists(os.path.join(file_dir, "enwik8.zip")):
        print(f"Dataset not found, downloading from {DATA_URL}")

        r = requests.get(DATA_URL, stream=True)
        total_size = int(r.headers.get("content-length", 0))

        with open(os.path.join(file_dir, "enwik8.zip"), "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    with zipfile.ZipFile(os.path.join(file_dir, "enwik8.zip"), "r") as zip_ref:
        zip_ref.extractall(file_dir)

    os.remove(os.path.join(file_dir, "enwik8.zip"))
    os.rename(os.path.join(file_dir, "enwik8"), input_file_path)


with open(input_file_path, "rb") as f:
    data = f.read()


def encode(s: bytes) -> list[int]:
    return list(s)


def decode(l: list[int]) -> bytes:
    return bytes(l)


# Assert that encode and decode are inverses
assert decode(encode(b"hello")) == b"hello"
assert encode(decode([104, 101, 108, 108, 111])) == [104, 101, 108, 108, 111]

n = len(data)
train_data = data[: int(n * SPLIT_RATIO)]
val_data = data[int(n * SPLIT_RATIO) :]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train data has {len(train_ids):,} tokens")
print(f"Validation data has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

meta = {"vocab_size": VOCAB_SIZE}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
