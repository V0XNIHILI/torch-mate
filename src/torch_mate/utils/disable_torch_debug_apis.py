import torch

def disable_torch_debug_apis():
    # Taken from: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
