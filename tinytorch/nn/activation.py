"""
Activation function modules for neural networks.

These are thin wrappers around the autograd Function classes,
providing a Module interface for use in Sequential and other containers.
"""

from ..autograd import (
    GELUFunction,
    ReLUFunction,
    SigmoidFunction,
    SoftmaxFunction,
    TanhFunction,
)
from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    """
    ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def forward(self, x: Tensor) -> Tensor:
        return ReLUFunction.apply(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation: f(x) = 1 / (1 + exp(-x))

    Maps any real number to (0, 1) range.
    Perfect for probabilities and binary classification.
    """

    def forward(self, x: Tensor) -> Tensor:
        return SigmoidFunction.apply(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def forward(self, x: Tensor) -> Tensor:
        return TanhFunction.apply(x)

    def __repr__(self) -> str:
        return "Tanh()"


class GELU(Module):
    """
    GELU activation: f(x) = x * Φ(x)

    Smooth approximation to ReLU, used in modern transformers.
    Where Φ(x) is the cumulative distribution function of standard normal.
    """

    def forward(self, x: Tensor) -> Tensor:
        return GELUFunction.apply(x)

    def __repr__(self) -> str:
        return "GELU()"


class Softmax(Module):
    """
    Softmax activation: f(x) = exp(x) / sum(exp(x))

    Converts logits to probability distribution.
    Output sums to 1 along the specified dimension.
    """

    def __init__(self, dim: int = -1):
        """
        Args:
            dim: Dimension along which to compute softmax (default: -1, last dim)
        """
        self.dim = dim

    def forward(self, x: Tensor, dim: int | None = None) -> Tensor:
        # Allow overriding dim at call time
        actual_dim = dim if dim is not None else self.dim
        return SoftmaxFunction.apply(x, actual_dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"
