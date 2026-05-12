import torch

def optimal_io_dtype(vocab_size):
        io_dtype = None
        for dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            if vocab_size <= torch.iinfo(dtype).max:
                io_dtype = dtype
                break
        if io_dtype is None:
            raise Exception(
                f"Vocabulary is too large to be represted as a uint: vocab_size={vocab_size}"
            )
        return io_dtype

