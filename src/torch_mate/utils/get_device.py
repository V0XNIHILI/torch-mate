import torch

from typing import List, Union

import os


def get_device(cuda_index: Union[List[int], int] = 0) -> torch.device:
    if isinstance(cuda_index, list):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(d) for d in cuda_index])

        device = torch.device('cuda')
    else:
        device = torch.device(
            f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")

    return device
