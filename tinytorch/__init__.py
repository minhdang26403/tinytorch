"""TinyTorch: a lightweight, educational deep learning toolkit."""

from . import nn, optim
from .tensor import Tensor

__all__ = ["Tensor", "nn", "optim"]

assert __all__ == sorted(__all__)
