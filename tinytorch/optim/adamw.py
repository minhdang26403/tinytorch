import numpy as np

from tinytorch.tensor import Tensor

from .optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    AdamW fixes a bug in Adam's weight decay implementation by decoupling
    weight decay from the gradient-based update. This leads to better
    regularization and is the preferred version for most applications.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        KEY DIFFERENCE from Adam:
        - Weight decay is applied directly to parameters, not added to gradients
        - This provides better regularization behavior
        """
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]

    def step(self):
        """
        Perform AdamW update step with decoupled weight decay.

        KEY DIFFERENCE from Adam:
        - Weight decay: θ_t = θ_t - lr * weight_decay * θ_t (applied after grad update)
        - NOT: grad = grad + weight_decay * param (Adam's incorrect approach)
        """
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            grad_data = grad.data if isinstance(grad, Tensor) else grad

            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(grad_data)
                self.v_buffers[i] = np.zeros_like(grad_data)

            m = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
            v = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * grad_data**2

            m_hat = m / (1 - self.beta1**self.step_count)
            v_hat = v / (1 - self.beta2**self.step_count)

            self.m_buffers[i] = m_hat
            self.v_buffers[i] = v_hat

            # Update parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply weight decay directly to the parameter
            if self.weight_decay != 0:
                param.data *= 1 - self.lr * self.weight_decay
