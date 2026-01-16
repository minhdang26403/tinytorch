import numpy as np

from ..tensor import Tensor

# Constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU


class Module:
    """
    Base class for all neural network modules (layers, containers, etc.).

    All modules should inherit from this class and implement:
    - forward(x): Compute layer output
    - parameters(): Return list of trainable parameters

    The __call__ method is provided to make layers callable.
    """

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            *args: Input tensors
            **kwargs: Additional arguments

        Returns:
            Output tensor after transformation
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Allow module to be called like a function.
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        """
        Return list of trainable parameters.

        Returns:
            List of Tensor objects (weights and biases)
        """
        return []  # Base class has no parameters

    def __repr__(self) -> str:
        """
        String representation of the module.
        """
        return f"{self.__class__.__name__}()"


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


class Sequential(Module):
    """
    Container that chains layers together sequentially.
    """

    def __init__(self, *layers: Module):
        """
        Initialize with layers to chain together.
        """
        # Accept both Sequential(layer1, layer2) and Sequential([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = tuple(layers[0])
        else:
            self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers sequentially.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        """
        Collect all parameters from all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"
