import torch

def disable_torch_debug_apis(anomaly=False, profiler=False, nvtx=False):
    # Disable Debug APIs for final fraining
     # Taken from: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
    torch.autograd.set_detect_anomaly(anomaly)
    torch.autograd.profiler.profile(profiler)
    torch.autograd.profiler.emit_nvtx(nvtx)

