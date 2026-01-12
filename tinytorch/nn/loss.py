import numpy as np

from ..tensor import Tensor

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax in a numerically stable way.
    """

    # Subtract input from max value to prevent overflow
    x_max = np.max(x.data, axis=dim, keepdims=True)
    x_shifted = x.data - x_max

    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=dim, keepdims=True))
    result_data = x.data - x_max - log_sum_exp

    return Tensor(result_data)


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error between predictions and targets.
        """
        error = predictions.data - targets.data
        squared_error = error**2
        mse = np.mean(squared_error)
        return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self, grad: Tensor) -> None:
        """
        Compute gradients.
        """
        pass


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and target class indices.
        """
        # Compute probabilities for each class
        log_probs = log_softmax(logits)

        batch_size = targets.shape[0]
        target_indices = targets.data.astype(int)

        # Select correct class probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        cross_entropy_loss = np.mean(-selected_log_probs)

        return Tensor(cross_entropy_loss)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)

    def backward(self, grad: Tensor) -> None:
        """
        Compute gradients.
        """
        pass


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.
        """

        # Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # Compute binary cross-entropy
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)
        bce_per_sample = -(
            targets.data * log_preds + (1 - targets.data) * log_one_minus_preds
        )

        # Return mean across all samples
        bce_loss = np.mean(bce_per_sample)
        return Tensor(bce_loss)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> None:
        """
        Compute gradients.
        """
        pass
