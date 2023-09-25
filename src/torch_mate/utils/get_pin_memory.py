import torch

def get_pin_memory(device: torch.device):
    if device.type == 'cuda':
        return True
    
    return False
