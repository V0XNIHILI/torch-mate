import torch

def disable_torch_debug_apis(anomaly=False, profiler=False, nvtx=False):
    """Disable PyTorch debug APIs for final training (after debugging).
    Code taken from: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b

    Args:
        anomaly (bool, optional): Whether to detect anomalies. Defaults to False.
        profiler (bool, optional): Whether to profile. Defaults to False.
        nvtx (bool, optional): Whether to emit nvtx. Defaults to False.
    """

    torch.autograd.set_detect_anomaly(anomaly)
    torch.autograd.profiler.profile(profiler)
    torch.autograd.profiler.emit_nvtx(nvtx)
