"""Random contiguous token batches for next-token prediction."""
import torch


def make_batches(ids, context_length, batch_size, device):
    data = torch.tensor(ids, dtype=torch.long)
    if len(data) <= context_length + 1:
        raise ValueError("Corpus is too small for the configured context length.")
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in starts])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


def split_ids(ids, train_fraction=0.9):
    cut = int(len(ids) * train_fraction)
    return ids[:cut], ids[cut:]
