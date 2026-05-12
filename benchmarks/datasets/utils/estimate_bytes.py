import torch

def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def estimate_sample_bytes(dataset):
    input_bytes  = _nbytes(dataset.data[0]['inputs'])
    output_bytes = _nbytes(dataset.data[0]['outputs'])
    mask_bytes   = _nbytes(dataset.data[0]['mask'])
    return input_bytes + output_bytes + mask_bytes

def compute_sample_bytes(sample):
    input_bytes  = _nbytes(sample['inputs'])
    output_bytes = _nbytes(sample['outputs'])
    mask_bytes   = _nbytes(sample['mask'])
    return input_bytes + output_bytes + mask_bytes
