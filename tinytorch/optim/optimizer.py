from tinytorch.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float = 0.0):
        """
        Initialize optimizer with parameters to optimize.
        """
        # Store parameters - gradient tracking is handled by autograd module
        self.params = params
        self.lr = lr
        self.step_count = 0  # For algorithms that need step counting

    def zero_grad(self):
        """
        Clear gradients from all parameters.
        """
        for param in self.params:
            param.grad = None

    def step(self):
        """
        Update parameters based on gradients.

        This is abstract - each optimizer implements its own update rule.
        """
        raise NotImplementedError("Subclasses must implement step()")
