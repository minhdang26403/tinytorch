"""
Neural network modules, activations, and losses for TinyTorch.
"""

from .activation import GELU, ReLU, Sigmoid, Softmax, Tanh
from .batchnorm import BatchNorm2d
from .conv import Conv2d
from .dropout import Dropout
from .embedding import (
    Embedding,
    EmbeddingLayer,
    PositionalEncoding,
    create_sinusoidal_embeddings,
)
from .linear import Linear
from .loss import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, log_softmax
from .module import Module
from .pooling import AvgPool2d, MaxPool2d
from .sequential import Sequential

__all__ = [
    "AvgPool2d",
    "BatchNorm2d",
    "BinaryCrossEntropyLoss",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "EmbeddingLayer",
    "GELU",
    "Linear",
    "MSELoss",
    "MaxPool2d",
    "Module",
    "PositionalEncoding",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "create_sinusoidal_embeddings",
    "log_softmax",
]

assert __all__ == sorted(__all__)
