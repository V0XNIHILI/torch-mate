import torch

class Sensitivity:
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        assert logits.dim() == 2, "Logits should be of shape (batch_size, 2)"

        preds = logits.argmax(dim=1)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
