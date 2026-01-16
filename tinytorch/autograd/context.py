"""
Context class for saving intermediate values during forward pass.

Similar to PyTorch's ctx object, this allows operations to save
tensors and metadata needed for backward computation.
"""


class Context:
    """
    Context for saving intermediate values during forward pass.

    Operations use this to save tensors and metadata that will be
    needed during the backward pass.
    """

    def __init__(self):
        """Initialize empty context."""
        self.saved_inputs = ()

    def save_for_backward(self, *inputs):
        """
        Save inputs needed for backward pass.

        Args:
            *inputs: Inputs to save for backward computation
        """
        self.saved_inputs = inputs
