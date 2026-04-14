import torch

class ToInt:
    def __call__(self, x):
        return int(torch.argmax(x))
