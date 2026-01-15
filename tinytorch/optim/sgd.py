from __future__ import annotations

from typing import Iterable, List

from ..tensor import Tensor


class SGD:
    """
    Minimal stochastic gradient descent optimizer placeholder.

    Gradient tracking is not implemented yet, but this keeps the public
    interface ready for when autograd support is added.
    """

    def __init__(self, params: Iterable[Tensor], lr: float = 0.01):
        self.params: List[Tensor] = list(params)
        self.lr = float(lr)

    def step(self) -> None:
        """
        Update parameters in-place (no-op until gradients are available).
        """
        # Intentionally left blank until grad fields are introduced.
        return None

    def zero_grad(self) -> None:
        """Reset gradients on all parameters (no-op placeholder)."""
        return None
