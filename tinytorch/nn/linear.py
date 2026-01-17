import numpy as np

from tinytorch.tensor import Tensor

from .module import Module

# Constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU


class Linear(Module):
    """
    Linear (fully connected) layer: y = xW + b

    This is the fundamental building block of neural networks.
    Applies a linear transformation to incoming data.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layer with Xavier weight initialization.
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization for stable gradients
        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data, requires_grad=True)

        # Initialize bias to zeros or None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through linear layer.
        """
        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output

    def parameters(self) -> list[Tensor]:
        """
        Return list of trainable parameters.
        """
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self) -> str:
        """
        String representation for debugging.
        """
        bias_str = f", bias={self.bias is not None}"
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}{bias_str})"
        )
