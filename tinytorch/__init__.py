"""
TinyTorch: a lightweight, educational deep learning toolkit.

This module provides the core components for building and training neural networks.
"""

from . import nn, optim
from .tensor import Tensor

__all__ = ["Tensor", "nn", "optim"]

assert __all__ == sorted(__all__)
