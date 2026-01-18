import numpy as np

from tinytorch.tensor import Tensor


class BatchNorm2d:
    """
    Batch Normalization for 2D spatial inputs (images).

    Normalizes activations across batch and spatial dimensions for each channel,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Key behaviors:
    - Training: Uses batch statistics, updates running statistics
    - Eval: Uses frozen running statistics for consistent inference

    Args:
        num_features: Number of channels (C in NCHW format)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics update (default: 0.1)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize BatchNorm2d layer.
        """

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (requires_grad=True for training)
        # gamma (scale): initialized to 1 so output = normalized input initially
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        # beta (shift): initialized to 0 so no shift initially
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not trained, accumulated during training)
        # These are used during evaluation for consistent normalization
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Training mode flag
        self.training = True

    def train(self):
        """Set layer to training mode."""
        self.training = True
        return self

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through BatchNorm2d.

        TODO: Implement batch normalization forward pass

        APPROACH:
        1. Validate input shape (must be 4D: batch, channels, height, width)
        2. If training:
           a. Compute batch mean and variance per channel
           b. Normalize using batch statistics
           c. Update running statistics with momentum
        3. If eval:
           a. Use running mean and variance
           b. Normalize using frozen statistics
        4. Apply scale (gamma) and shift (beta)

        EXAMPLE:
        >>> bn = BatchNorm2d(16)
        >>> x = Tensor(np.random.randn(2, 16, 8, 8))  # batch=2, channels=16, 8x8
        >>> y = bn(x)
        >>> print(y.shape)  # (2, 16, 8, 8) - same shape

        HINTS:
        - Compute mean/var over axes (0, 2, 3) to get per-channel statistics
        - Reshape gamma/beta to (1, C, 1, 1) for broadcasting
        - Running stat update: running = (1 - momentum) * running + momentum * batch
        """

        # Input validation
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.shape}"
            )

        _, channels, _, _ = x.shape

        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {channels}")

        if self.training:
            batch_mean = np.mean(x.data, axis=(0, 2, 3))
            batch_var = np.var(x.data, axis=(0, 2, 3))
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize: (x - mean) / sqrt(var + eps)
        # Reshape mean and var for broadcasting: (C,) -> (1, C, 1, 1)
        mean_reshaped = mean.reshape(1, channels, 1, 1)
        var_reshaped = var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

        # Apply scale (gamma) and shift (beta)
        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        output = gamma_reshaped * x_normalized + beta_reshaped

        # Return Tensor with gradient tracking
        requires_grad = (
            x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        )
        result = Tensor(output, requires_grad=requires_grad)

        return result

    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)
