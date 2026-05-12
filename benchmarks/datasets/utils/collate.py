import torch
from typing import Union, Sequence, List, Tuple, Dict
from functools import partial
from torch.utils.data import Dataset, DataLoader


def collate_fn(
    max_len,
    pad_id,
    input_dtype,
    output_dtype,
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length sequences in a batch.
    """

    def pad_1d(t: torch.Tensor, length: int, value: int) -> torch.Tensor:
        if t.size(0) == length:
            return t
        out = t.new_full((length,), value)
        out[: t.size(0)] = t
        return out

    inputs  = []
    mask    = []
    outputs = []
    stats   = []
    for seq in batch:
        inputs.append(
                pad_1d(seq["inputs"], max_len, pad_id)
                )
        outputs.append(
                pad_1d(seq["outputs"], max_len, pad_id)
                )
        mask.append(
                pad_1d(seq["mask"], max_len, False)
                )
        stats.append(
                seq["stats"]
                )
    return {
            "inputs":  torch.stack(inputs,  dim=0).to(dtype=input_dtype),
            "outputs": torch.stack(outputs, dim=0).to(dtype=output_dtype),
            "mask":    torch.stack(mask,    dim=0).to(dtype=torch.bool),
            "stats":   stats
            }

def make_collate_fn(max_len, pad_id, input_dtype, output_dtype):
    return partial(collate_fn, max_len, pad_id, input_dtype, output_dtype)


def merge_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Merge batches
    return {sample['idx']: sample for sample in batch}

def make_merge_collate_fn():
    return merge_collate_fn

