from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from minillm.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class Dataset:
    train: np.ndarray  # int64 tokens
    val: np.ndarray


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def make_dataset(text: str, *, tokenizer: ByteTokenizer, val_fraction: float = 0.05) -> Dataset:
    ids = tokenizer.encode(text) + [tokenizer.eos_id]
    arr = np.array(ids, dtype=np.int64)
    n = len(arr)
    split = int(n * (1.0 - val_fraction))
    return Dataset(train=arr[:split], val=arr[split:])


def get_batch(data: np.ndarray, *, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(data)
    if n <= block_size + 1:
        raise ValueError(f"Data too short for block_size={block_size}: need {block_size + 2} tokens, got {n}")
    ix = np.random.randint(0, n - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    return x_t, y_t
