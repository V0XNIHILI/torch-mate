import torch

from typing import List, Union

import os


def get_device(cuda_index: Union[List[int], int] = 0, hide_others: bool = True) -> torch.device:
    """Get one or more CUDA devices. In case CUDA is not available, either MPS (Apple MX machines) or the CPU is used.

    Args:
        cuda_index (Union[List[int], int], optional): Which CUDA device index/indices to use. Defaults to 0.
        hide_others (bool, optional): Whether to hide other CUDA devices that are not getted. Defaults to True.

    Returns:
        torch.device: The CUDA device(s) to use.
    """

    if torch.cuda.is_available():
        if not isinstance(cuda_index, list):
            cuda_index = [cuda_index]

        if hide_others or len(cuda_index) > 1:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(d) for d in cuda_index])

            device = 'cuda'
        else:
            device = f"cuda:{cuda_index[0]}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)
