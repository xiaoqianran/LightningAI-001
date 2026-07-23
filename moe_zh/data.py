from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBinDataset(Dataset):
    """Memory-mapped token id stream; each item is a (block_size,) window."""

    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        # uint32 tokens
        self.data = np.memmap(self.bin_path, dtype=np.uint32, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"{bin_path} too short ({len(self.data)} tokens); need > {block_size}"
            )

    def __len__(self) -> int:
        # number of random-access starting positions
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # map idx into range (Dataset may sample any int with replacement in train loop)
        idx = int(idx) % len(self)
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def write_token_bin(ids: list[int], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(ids, dtype=np.uint32)
    arr.tofile(path)
    return int(arr.size)
