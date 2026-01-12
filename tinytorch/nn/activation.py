import numpy as np

from ..tensor import Tensor


class Sigmoid:
    """
    Sigmoid activation

    Maps any real number to (0, 1) range.
    Perfect for probabilities and binary classification.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sigmoid activation element-wise.
        """
        # Clip extreme values to prevent overflow (sigmoid(-500) ~ 0, sigmoid(500) ~ 1)
        # Clipping activations to (-500, 500) ensures exp() stays within float64 range
        z = np.clip(x.data, -500, 500)

        # Use numerically stable sigmoid
        # For positive values: 1 / (1 + exp(-x))
        # For negative values: exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)) after clipping
        result_data = np.zeros_like(z)

        # Positive values (including zero)
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Negative values
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Compute gradient."""
        pass


class ReLU:
    """
    ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation element-wise."""
        return Tensor(np.maximum(x.data, 0))

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Compute gradient."""
        pass


class Tanh:
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation element-wise."""
        return Tensor(np.tanh(x.data))

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Compute gradient."""
        pass


class GELU:
    """
    GELU activation: f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)

    Smooth approximation to ReLU, used in modern transformers.
    Where Φ(x) is the cumulative distribution function of standard normal.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation element-wise."""
        # GELU approximation: x * sigmoid(1.702 * x)
        # First compute sigmoid part
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        # Then multiply by x
        result = x.data * sigmoid_part
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Compute gradient."""
        pass


class Softmax:
    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Apply softmax activation along specified dimension.
        """
        # Numerical stability: subtract max to prevent overflow
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max  # Tensor subtraction

        # Compute exponentials
        exp_values = np.exp(x_shifted)

        # Sum along dimension
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        pass
