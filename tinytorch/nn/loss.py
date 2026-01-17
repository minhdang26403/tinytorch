"""
Loss function modules for neural networks.

These are thin wrappers around the autograd Function classes,
providing a Module interface consistent with PyTorch's loss functions.
"""

import numpy as np

from tinytorch.autograd import (
    BinaryCrossEntropyFunction,
    CrossEntropyFunction,
    MSEFunction,
)
from tinytorch.tensor import Tensor

from .module import Module


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax in a numerically stable way.

    Args:
        x: Input tensor
        dim: Dimension along which to compute log-softmax

    Returns:
        Tensor with log-softmax applied
    """
    # Subtract max for numerical stability
    x_max = np.max(x.data, axis=dim, keepdims=True)
    x_shifted = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=dim, keepdims=True))
    result_data = x_shifted - log_sum_exp
    return Tensor(result_data, requires_grad=x.requires_grad)


class MSELoss(Module):
    """
    Mean Squared Error loss for regression tasks.

    Computes: mean((predictions - targets)^2)
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return MSEFunction.apply(predictions, targets)

    def __repr__(self) -> str:
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for multi-class classification.

    Combines softmax and negative log-likelihood in a numerically stable way.
    Takes raw logits (unnormalized scores) and target class indices.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return CrossEntropyFunction.apply(logits, targets)

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"


class BinaryCrossEntropyLoss(Module):
    """
    Binary cross-entropy loss for binary classification.

    Takes predictions (after sigmoid, in range [0, 1]) and binary targets.
    Computes: -mean(target * log(pred) + (1 - target) * log(1 - pred))
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return BinaryCrossEntropyFunction.apply(predictions, targets)

    def __repr__(self) -> str:
        return "BinaryCrossEntropyLoss()"
