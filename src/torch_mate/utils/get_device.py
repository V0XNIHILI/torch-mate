import torch

from typing import List, Union

import os


def get_device(cuda_index: Union[List[int], int] = 0) -> torch.device:
    """Get one or more CUDA devices. In case CUDA is not available, the CPU is used.

    Args:
        cuda_index (Union[List[int], int], optional): Which CUDA device index/indices to use. Defaults to 0.

    Returns:
        torch.device: The CUDA device(s) to use.
    """

    if torch.cuda.is_available():
        if isinstance(cuda_index, list):
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(d) for d in cuda_index])

            device = 'cuda'
        else:
            device = f"cuda:{cuda_index}"
    else:
        device = "cpu"

    return torch.device(device)
