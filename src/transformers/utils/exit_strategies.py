
import torch


class ConfidenceThreshold():
    def __init__(self, threshold = 1.0, *args, **kwargs):
        self.exit_threshold = threshold

    def __call__(self, logits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert (logits.shape[0] == 1)  # make sure we are running inference with batch of size 1
        probs = torch.softmax(logits, dim=-1)
        return torch.max(probs, dim=1)[0] >= self.exit_threshold





EXIT_MAP = {
    'confidence_threshold': ConfidenceThreshold,
}