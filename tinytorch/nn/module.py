from tinytorch.tensor import Tensor


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
