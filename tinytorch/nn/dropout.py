import numpy as np

from ..tensor import Tensor
from .module import Module

# Constants for dropout probabilities
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Dropout(Module):
    """
    Dropout layer for regularization.

    During training, randomly zeros elements with probability p, scales survivors
    by 1/(1-p). During inference, passes input through unchanged.
    """

    def __init__(self, p=0.5):
        """
        Initialize dropout layer.

        Args:
            p: Probability of zeroing each element
        """
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Dropout probability must be between {DROPOUT_MIN_PROB} and "
                f"{DROPOUT_MAX_PROB}, got {p}"
            )
        self.p = p

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Forward pass through dropout layer.

        During training, randomly zeros elements with probability p, scales survivors
        by 1/(1-p). During inference, passes input through unchanged.

        Prevents overfitting by forcing the network to not rely on specific neurons.
        """
        if not training or self.p == DROPOUT_MIN_PROB:
            return x

        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))

        keep_prob = 1 - self.p
        mask = np.random.random(x.shape) < keep_prob
        mask_tensor = Tensor(mask.astype(np.float32))
        scale = 1 / keep_prob
        return x * mask_tensor * scale
