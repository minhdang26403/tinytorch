import numpy as np

from tinytorch.tensor import Tensor

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    SGD is the foundational optimization algorithm that moves parameters
    in the direction opposite to gradients. With momentum, it remembers
    previous updates to reduce oscillations and accelerate convergence.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        Initialize SGD optimizer.
        """
        super().__init__(params, lr)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        """
        Check if this optimizer uses momentum.

        This explicit API method replaces the need for hasattr() checks
        in checkpointing code (Module 08).

        Returns:
            bool: True if momentum is enabled (momentum > 0), False otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.has_momentum()
            True
        """
        return self.momentum > 0

    def get_momentum_state(self) -> list | None:
        """
        Get momentum buffers for checkpointing.

        This explicit API method provides safe access to momentum buffers
        without using hasattr(), making the API contract clear.

        Returns:
            Optional[List]: List of momentum buffers if momentum is enabled,
                          None otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.step()  # Initialize buffers
            >>> state = optimizer.get_momentum_state()
            >>> # Later: optimizer.set_momentum_state(state)
        """
        if not self.has_momentum():
            return None
        return [
            buf.copy() if buf is not None else None for buf in self.momentum_buffers
        ]

    def set_momentum_state(self, state: list | None) -> None:
        """
        Restore momentum buffers from checkpointing.

        This explicit API method provides safe restoration of momentum state
        without using hasattr().

        Args:
            state: List of momentum buffers or None

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> state = optimizer.get_momentum_state()
            >>> # Training interruption...
            >>> new_optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> new_optimizer.set_momentum_state(state)
        """
        if state is None or not self.has_momentum():
            return

        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"State length {len(state)} doesn't match "
                f"optimizer parameters {len(self.momentum_buffers)}"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        """
        Perform SGD update step with momentum.
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient data - grad can be Tensor or numpy array
            grad = param.grad
            grad_data = grad.data if isinstance(grad, Tensor) else grad

            if self.weight_decay != 0:
                grad_data += self.weight_decay * param.data

            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(grad_data)

                self.momentum_buffers[i] = (
                    self.momentum * self.momentum_buffers[i] + grad_data
                )
                grad_data = self.momentum_buffers[i]

            param.data -= self.lr * grad_data

        self.step_count += 1
