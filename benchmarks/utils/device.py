import torch


# Set device: cuda or cpu (preference in this order)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pin_memory = torch.cuda.is_available()

