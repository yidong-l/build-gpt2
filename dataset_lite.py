import bisect
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class FineWebDataset(Dataset):
    def __init__(self, data_dir, file_prefix= "edufineweb", split="train", T=1024):
        """
        Args:
            data_dir: Path to the directory containing .npy files (e.g., 'edu_fineweb10B')
            file_prefix: The prefix of the .npy files (e.g., 'edufineweb')
            split: 'train' or 'val' to select the correct shards.
            T: The context length for the model (e.g., 1024 tokens).
        """
        self.data_dir = data_dir
        self.T = T

        # 1. Search for all relevant shards
        pattern = os.path.join(data_dir, f"edufineweb_{split}_*.npy")
        self.shards = sorted(glob.glob(pattern))

        if len(self.shards) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir} for split '{split}'")


        print(f"Loading {split} dataset with {len(self.shards)} shards...")

        # 2. Pre-calculate the number of tokens available in each shard.
        # and keep memory-mapped versions of each shard.
        #    We do this once at initialization to keep __getitem__ fast.
        self.shard_offsets = [0]
        self.mmaps = []

        num_tokens = 0
        for p in self.shards:
            # We use mmap_mode='r' to read the header without loading the data
            # This is fast even for large files.
            mmap = np.load(p, mmap_mode='r')
            num_tokens += len(mmap)
            self.shard_offsets.append(num_tokens)
            self.mmaps.append(mmap)

        self.num_tokens = num_tokens
        self.len = (self.num_tokens - 1) // T

        print(f"Found {len(self.shards)} shards, total tokens = {self.num_tokens}, total samples: {self.len} of size {T}.")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx * self.T
        end = start + self.T + 1

        tensor_np = self._get_tokens_in_range(start, end)
        tensor_pt = torch.tensor(tensor_np, dtype=torch.long)

        # Note: We must convert uint16 -> int64 (long) for PyTorch Embedding layers
        # x, y shifted by 1
        x = tensor_pt[:-1]
        y = tensor_pt[1:]

        return x, y

    def _get_tokens_in_range(self, start, end):
        """
        Returns ndarray, shape=(shard_size,); dtype=uint16
        """
        # Find which shard contains the start token
        shard_idx = bisect.bisect_right(self.shard_offsets, start) - 1


        # Get the tokens from that shard
        mmap = self.mmaps[shard_idx]
        local_start = start - self.shard_offsets[shard_idx]
        local_end = end - self.shard_offsets[shard_idx]

        # Case 1: The requested range is fully within this single shard
        if local_end <= len(mmap):
            return mmap[local_start:local_end]

        # Case 2: The requested range spans across two shards
        mmap2 = self.mmaps[shard_idx + 1]
        part1 = mmap[local_start:]
        remainder = local_end - len(mmap)
        assert remainder <= len(mmap2), f"Requested range {start} - {end} span more than two shards: {shard_idx} and {shard_idx + 1}"

        part2 = mmap2[:local_end - len(mmap)]

        return np.concatenate([part1, part2])

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, data_dir, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_dir = data_dir
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = self.data_dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
