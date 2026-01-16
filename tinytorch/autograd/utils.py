"""
Utility functions for autograd operations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


def unbroadcast(grad: "Tensor", target_shape: tuple[int, ...]) -> "Tensor":
    """
    Reduce gradient to match a broadcasted input's shape.

    Args:
        grad: Gradient tensor that may have been broadcasted
        target_shape: Target shape to reduce to

    Returns:
        Gradient tensor with target shape
    """
    if grad.shape == target_shape:
        return grad

    g = grad
    # Remove leading broadcasted dims
    while len(g.shape) > len(target_shape):
        g = g.sum(axis=0)

    # Sum over axes where target dim was 1 (broadcasted)
    for i, dim in enumerate(target_shape):
        if dim == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)

    return g
