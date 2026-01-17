import numpy as np

from tinytorch.tensor import Tensor

from .optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer with adaptive learning rates.

    RMSProp (Root Mean Square Propagation) adapts the learning rate for each
    parameter by dividing by a running average of recent gradient magnitudes.
    This helps handle non-stationary objectives and works well for RNNs.

    Unlike Adam, RMSProp does not include bias correction, making it simpler
    but potentially less stable in the early stages of training.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize RMSProp optimizer.

        PARAMETERS:
        - lr: Learning rate (default: 0.01)
        - alpha: Smoothing constant for running average (default: 0.99)
        - eps: Small constant for numerical stability (default: 1e-8)
        - weight_decay: L2 penalty coefficient (default: 0.0)

        EXAMPLE:
            >>> params = [Tensor([1.0, 2.0], requires_grad=True)]
            >>> optimizer = RMSProp(params, lr=0.01, alpha=0.99)
            >>> # Training loop
            >>> optimizer.step()
        """
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        # Initialize squared gradient buffers (created lazily)
        self.v_buffers = [None for _ in params]

    def step(self):
        """
        Perform RMSProp update step.

        APPROACH:
        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update running average of squared gradients
           c. Update parameter using adaptive learning rate

        FORMULAS:
        - v_t = α * v_{t-1} + (1-α) * g_t²
        - θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)

        Where:
        - v_t: Running average of squared gradients
        - α: Smoothing constant (alpha)
        - g_t: Current gradient
        - ε: Small constant for numerical stability
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            grad_data = grad.data if isinstance(grad, Tensor) else grad

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Initialize buffer on first use
            if self.v_buffers[i] is None:
                self.v_buffers[i] = np.zeros_like(grad_data)

            # Update running average of squared gradients
            v = self.alpha * self.v_buffers[i] + (1 - self.alpha) * grad_data**2
            self.v_buffers[i] = v

            # Update parameter with adaptive learning rate
            param.data -= self.lr * grad_data / (np.sqrt(v) + self.eps)

        self.step_count += 1
