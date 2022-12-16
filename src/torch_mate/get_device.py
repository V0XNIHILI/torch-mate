import torch

def get_device(cuda_index: int=0):
    device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")

    return device