from dataset_lite import FineWebDataset, DataLoaderLite
from torch.utils.data import DataLoader, Dataset
import unittest
import numpy as np
import torch

class TestFineWebDataset(unittest.TestCase):
    DATA_DIR = '/home/ubuntu/local_data/edu_fineweb10B/'

    def test_initialization(self):
        dataset = FineWebDataset(data_dir=self.DATA_DIR, split='train', T=1024)
        self.assertGreater(len(dataset.shards), 0)
        self.assertGreater(dataset.num_tokens, 0)
        self.assertGreater(len(dataset), 0)

    def test_getitem(self):
        dataset = FineWebDataset(data_dir=self.DATA_DIR, split='train', T=1024)
        x, y = dataset[0]
        self.assertEqual(x.shape[0], 1024)
        self.assertEqual(y.shape[0], 1024)
        self.assertTrue(torch.all(x[1:] == y[:-1]))

    def test_read_across_shards(self):
        B = 64
        T = 1024
        NUM_SHARDS = 3
        standard_dataset = FineWebDataset(
            data_dir=self.DATA_DIR, split='train', T=T)
        standard_loader = DataLoader(
            standard_dataset, batch_size=B, shuffle=False, drop_last=True)
        merged = np.concatenate(standard_dataset.mmaps[:NUM_SHARDS])

        loop = (len(merged) - 1) // (B * T)
        std_iter = iter(standard_loader)
        for i in range(loop):
            x_std, y_std = next(std_iter)
            start = i * B * T
            end = start + B * T + 1
            buf = merged[start:end]
            x_expected = torch.tensor(buf[:-1], dtype=torch.long).view(B, T)
            y_expected = torch.tensor(buf[1:], dtype=torch.long).view(B, T)

            self.assertTrue(torch.equal(x_std, x_expected), msg=f"Mismatch in x at batch {i}")
            self.assertTrue(torch.equal(y_std, y_expected), msg=f"Mismatch in y at batch {i}")

    def test_dataloader_equivalency(self):
        B = 64
        lite_loader = DataLoaderLite(
            data_dir=self.DATA_DIR, split='train',
            B=B, T=1024,
            process_rank=0, num_processes=1)

        standard_dataset = FineWebDataset(
            data_dir=self.DATA_DIR, split='train', T=1024)
        standard_loader = DataLoader(
            standard_dataset, batch_size=B, shuffle=False, drop_last=True)

        # standard_loader attempts to read across shards while lite_loader does not.
        # So we only compare batches from the first shard file.
        loop = 1524
        std_iter = iter(standard_loader)
        for i in range(loop):
            x_std, y_std = next(std_iter)
            x_lite, y_lite = lite_loader.next_batch()

            self.assertTrue(torch.equal(x_std, x_lite), msg=f"Mismatch in x at batch {i}")
            self.assertTrue(torch.equal(y_std, y_lite), msg=f"Mismatch in y at batch {i}")

