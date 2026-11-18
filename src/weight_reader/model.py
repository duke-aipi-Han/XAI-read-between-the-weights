"""Small, inspectable network and its known calibration circuit."""
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class WeightReaderMLP(nn.Module):
    """4→8→8→1 ReLU MLP. Linear weights use output-by-input orientation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor, ablate: Iterable[int] = ()) -> torch.Tensor:
        return self.activations(x, ablate)[2]

    def activations(self, x: torch.Tensor, ablate: Iterable[int] = ()):
        """Return first and second hidden activations and logits, preserving batches."""
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError('inputs must have shape (batch, 4)')
        ids = tuple(ablate)
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= 8 for i in ids):
            raise ValueError('ablation neuron IDs must be integers in 0..7')
        if len(set(ids)) != len(ids):
            raise ValueError('ablation neuron IDs must be unique')
        h1 = F.relu(self.fc1(x))
        if ids:
            mask = torch.ones(8, dtype=h1.dtype, device=h1.device)
            mask[list(ids)] = 0
            h1 = h1 * mask
        h2 = F.relu(self.fc2(h1))
        return h1, h2, self.fc3(h2)


def calibration_model() -> WeightReaderMLP:
    """Return the hand-set same-sign circuit; unused units have zero effect."""
    model = WeightReaderMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc1.weight[:4, :2] = torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
        model.fc2.weight[:4, :4] = torch.eye(4)
        model.fc3.weight[0, :4] = torch.tensor([5., 5., -5., -5.])
    return model
