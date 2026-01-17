import numpy as np

from tinytorch.tensor import Tensor

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.

    Adam computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.
    This makes it effective for problems with sparse gradients or noisy data.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adam optimizer.

        PARAMETERS:
        - lr: Learning rate (default: 0.001)
        - betas: Coefficients for computing running averages (default: (0.9, 0.999))
        - eps: Small constant for numerical stability (default: 1e-8)
        - weight_decay: L2 penalty coefficient (default: 0.0)
        """
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)

    def step(self):
        """
        Perform Adam update step.

        TODO: Implement Adam parameter update with adaptive learning rates

        APPROACH:
        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update first moment estimate (momentum of gradient)
           c. Update second moment estimate (momentum of squared gradient)
           d. Compute bias-corrected moments
           e. Update parameter using adaptive learning rate

        FORMULAS:
        - m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        - v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
        - m̂_t = m_t / (1-β₁^t)
        - v̂_t = v_t / (1-β₂^t)
        - θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

        HINTS:
        - Initialize buffers as zeros on first use
        - Use step_count for bias correction
        - Square gradients element-wise for second moment
        """
        # Increment step counter first (needed for bias correction)
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            grad_data = grad.data if isinstance(grad, Tensor) else grad

            if self.weight_decay != 0:
                grad_data += self.weight_decay * param.data

            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(grad_data)
                self.v_buffers[i] = np.zeros_like(grad_data)

            # Calculate first moment and second moment estimate of gradient
            m = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
            v = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * grad_data**2

            # Compute bias-corrected moments
            m_hat = m / (1 - self.beta1**self.step_count)
            v_hat = v / (1 - self.beta2**self.step_count)

            # Update m and v
            self.m_buffers[i] = m_hat
            self.v_buffers[i] = v_hat

            # Update parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
