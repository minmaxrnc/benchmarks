import torch

def bytes2miB(x, fmt='{:.1f}'):
    return fmt.format(x / (1024**2)) + 'MiBs'

def gpu_memory(key):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)   # driver-level
    reserved = torch.cuda.memory_reserved(device)               # managed by PyTorch
    allocated = torch.cuda.memory_allocated(device)             # actually used by tensors

    # Memory PyTorch could likely allocate right now (reuse its cache + driver-free)
    available_for_torch = free_bytes + (reserved - allocated)
    available_for_torch = max(0, min(available_for_torch, total_bytes))

    res = {
        "total_bytes": total_bytes,
        "free_bytes_driver": free_bytes,
        "used_bytes_driver": total_bytes - free_bytes,
        "reserved_bytes_torch": reserved,
        "allocated_bytes_torch": allocated,
        "available_bytes_for_torch": available_for_torch,  # heuristic upper bound
    }
    return res[key]

