"""
FineWeb-Edu dataset

Downloads and tokenizes the data and saves data shards to disk.

Run as:
$ python fineweb.py
"""

import argparse
import os
import multiprocessing as mp
import numpy as np
import tiktoken
import time
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./data/edu_fineweb10B', help='Directory to save the tokenized data shards.')
args = parser.parse_args()
DATA_CACHE_DIR = args.output

remote_name = "sample-10BT"

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

def write_datafile(filename, tokens_np):
    """Writes a numpy array of uint16 tokens to a binary file."""
    with open(filename, 'wb') as f:
        f.wrote(tokens_np.tobytes())

if __name__ == "__main__":
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
