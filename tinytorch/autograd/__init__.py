"""
Autograd package for automatic differentiation.

This package implements the autograd engine following PyTorch's Function pattern,
where each operation encapsulates both forward and backward logic.
"""

from .context import Context
from .function import (
    AccumulateGrad,
    AddFunction,
    BinaryCrossEntropyFunction,
    CrossEntropyFunction,
    DivFunction,
    ExpFunction,
    Function,
    GELUFunction,
    MatmulFunction,
    MaxFunction,
    MeanFunction,
    MSEFunction,
    MulFunction,
    PowFunction,
    ReLUFunction,
    ReshapeFunction,
    SigmoidFunction,
    SoftmaxFunction,
    SubFunction,
    SumFunction,
    TanhFunction,
    TransposeFunction,
)

__all__ = [
    "AccumulateGrad",
    "AddFunction",
    "BinaryCrossEntropyFunction",
    "Context",
    "CrossEntropyFunction",
    "DivFunction",
    "ExpFunction",
    "Function",
    "GELUFunction",
    "MSEFunction",
    "MatmulFunction",
    "MaxFunction",
    "MeanFunction",
    "MulFunction",
    "PowFunction",
    "ReLUFunction",
    "ReshapeFunction",
    "SigmoidFunction",
    "SoftmaxFunction",
    "SubFunction",
    "SumFunction",
    "TanhFunction",
    "TransposeFunction",
]

assert __all__ == sorted(__all__)
