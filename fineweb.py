"""
FineWeb-Edu dataset

Downloads and tokenizes the data and saves data shards to disk.

Run as:
$ python fineweb.py
"""
import argparse
import os
import sys
import multiprocessing as mp
import numpy as np
import tiktoken
import time
from datasets import load_dataset
from tqdm import tqdm

remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards.

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./data/edu_fineweb10B', help='Directory to save the tokenized data shards.')
parser.add_argument('--hf_cache', type=str, help='HuggingFace datasets cache directory.')
args = parser.parse_args()
DATA_CACHE_DIR = args.output
HF_DATASETS_CACHE_DIR = args.hf_cache


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

class ShardWriter:
    """
    Buffered writer for tokenized data.
    Buffers tokens in memory buffer[shard_size] and flushes to sharded .npy files.

    Similar to io.BufferedWriter (python) and bufio.Writer (golang).
    """

    def __init__(self, data_cache_dir, shard_size):
        self.data_cache_dir = data_cache_dir
        self.buf = np.empty((shard_size,), dtype=np.uint16)
        self.buf_len = 0
        self.buf_cap = shard_size
        self.shard_idx = 0
        self.progress_bar = None

    def _filename(self):
        split = "val" if self.shard_idx == 0 else "train"
        filename = os.path.join(self.data_cache_dir, f"edufineweb_{split}_{self.shard_idx:06d}")
        return filename

    def write(self, tokens):
        """Writes tokens to the buffer, flushing to disk when full."""

        # Use a while loop to keep copying tokens to buffer and flushing when full.
        copied = 0
        while copied < len(tokens):
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=self.buf_cap, unit="tokens", desc=f"Shard {self.shard_idx}")
 
            # how many spaces left vs. how many remainin tokens to copy
            batch = min(self.buf_cap - self.buf_len, len(tokens) - copied)

            self.buf[self.buf_len : self.buf_len + batch] = tokens[copied : copied + batch]

            self.buf_len += batch
            copied += batch
            self.progress_bar.update(batch)

            if self.buf_len == self.buf_cap:
                self.flush()

    def flush(self):
        """Writes the buffer to disk and resets the buffer."""
        if self.buf_len == 0:
            return

        filename = self._filename()
        np.save(filename, self.buf[:self.buf_len])

        self.shard_idx += 1
        self.buf_len = 0

        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None


if __name__ == "__main__":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir=HF_DATASETS_CACHE_DIR)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    writer = ShardWriter(DATA_CACHE_DIR, shard_size)

    # Multi tokenizers processes and single writer (main) process
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            writer.write(tokens)
 
    writer.flush()
