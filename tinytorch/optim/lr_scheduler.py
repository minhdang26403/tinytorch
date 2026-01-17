import numpy as np

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule


class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr, decreases following a cosine curve to min_lr over T epochs.
    This provides aggressive learning initially, then fine-tuning at the end.

    TODO: Implement cosine annealing schedule

    APPROACH:
    1. Store max_lr, min_lr, and total_epochs
    2. In get_lr(), compute cosine factor: (1 + cos(π * epoch / total_epochs)) / 2
    3. Interpolate: min_lr + (max_lr - min_lr) * cosine_factor

    EXAMPLE:
    >>> schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
    >>> print(schedule.get_lr(0))    # Start: 0.1
    >>> print(schedule.get_lr(50))   # Middle: ~0.055
    >>> print(schedule.get_lr(100))  # End: 0.01
    """

    def __init__(
        self,
        max_lr: float = DEFAULT_MAX_LR,
        min_lr: float = DEFAULT_MIN_LR,
        total_epochs: int = DEFAULT_TOTAL_EPOCHS,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        return (
            self.min_lr
            + (self.max_lr - self.min_lr)
            * (1 + np.cos(np.pi * epoch / self.total_epochs))
            / 2
        )
