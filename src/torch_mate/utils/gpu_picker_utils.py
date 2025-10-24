from typing import Optional
import os
import subprocess

import torch


def get_gpu_memory_usage():
    """Get GPU memory usage using nvidia-smi."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'])
        gpu_memory_info = output.decode('utf-8').strip().split('\n')
        gpu_memory_usage = [(int(info.split(',')[0]), int(info.split(',')[1])) for info in gpu_memory_info]

        return gpu_memory_usage
    except subprocess.CalledProcessError:
        print("Error: nvidia-smi command failed.")

        return None


def sort_gpus_by_memory_usage(limit_gpu_ids: Optional[list] = None):
    """Sort GPUs by memory usage and return a list of GPU IDs.

    Args:
        limit_gpu_ids (Optional[list]): List of GPU IDs to consider for sorting. Defaults to None.

    Returns:
        Union[list, None]: List of sorted GPU IDs or None if nvidia-smi command failed.
    """
    gpu_memory_usage = get_gpu_memory_usage()

    if gpu_memory_usage is None:
        raise RuntimeError("nvidia-smi command failed or no GPUs available.")
    
    sorted_gpus = sorted(gpu_memory_usage, key=lambda x: x[1])
    sorted_gpu_ids = [gpu[0] for gpu in sorted_gpus]

    return list(filter(lambda x: x in limit_gpu_ids, sorted_gpu_ids)) if limit_gpu_ids else sorted_gpu_ids


def make_only_least_used_gpu_visible(limit_gpu_ids: Optional[list] = None):
    """Make only the least used GPU visible to the current process.

    Args:
        limit_gpu_ids (Optional[list]): List of GPU IDs to consider for sorting. Defaults to None.
    """
    sorted_gpu_ids = sort_gpus_by_memory_usage(limit_gpu_ids)

    if sorted_gpu_ids is None:
        return None

    if len(sorted_gpu_ids) > 0:
        visible_gpu_id = sorted_gpu_ids[0]
        print("Setting visible GPU ID:", visible_gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_id)
    else:
        print("No GPUs available for the current process.")


def get_least_used_gpu(limit_gpu_ids: Optional[list] = None):
    """Get the least used GPU ID.

    Args:
        limit_gpu_ids (Optional[list]): List of GPU IDs to consider for sorting. Defaults to None.
    """
    sorted_gpu_ids = sort_gpus_by_memory_usage(limit_gpu_ids)

    if sorted_gpu_ids is None:
        return torch.device("cpu")
    
    if len(sorted_gpu_ids) > 0:
        make_only_least_used_gpu_visible(limit_gpu_ids)

        return torch.device("cuda")


def get_nvlink_topology():
    try:
        output = subprocess.check_output(['nvidia-smi', 'topo', '-m'])
        gpu_memory_info = output.decode('utf-8').strip().split('\n')
        lines = gpu_memory_info
    except subprocess.CalledProcessError:
        print("Error: nvidia-smi command failed.")

        return None
    
    lines = [line.split() for line in lines]
    num_gpus = len(lines[0]) - 4
    conns = [[i - 1 for i, value in enumerate(line) if value.startswith("NV")] for line in lines[1:num_gpus+1]]

    return conns
