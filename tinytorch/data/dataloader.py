from typing import Iterator

import numpy as np

from tinytorch.tensor import Tensor

from .dataset import Dataset


class DataLoader:
    """
    Data loader with batching and shuffling support.

    Wraps a dataset to provide batched iteration with optional shuffling.
    Essential for efficient training with mini-batch gradient descent.

    EXAMPLE:
    >>> dataset = TensorDataset(Tensor([[1,2], [3,4], [5,6]]), Tensor([0,1,0]))
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     features_batch, labels_batch = batch
    ...     print(f"Features: {features_batch.shape}, Labels: {labels_batch.shape}")
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle=False):
        """
        Create DataLoader for batched iteration.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """
        Return number of batches per epoch.

        EXAMPLE:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4], [5]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> print(len(loader))  # 3 (batches: [2, 2, 1])
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """
        Return iterator over batches.

        EXAMPLE:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> for batch in loader:
        ...     print(batch[0].shape)  # (2, 1)
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate batch - convert list of tuples to tuple of tensors
            yield self._collate_batch(batch)

    def _collate_batch(self, batch: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        """
        Collate individual samples into batched tensors.

        Args:
            batch: List of sample tuples from dataset

        Returns:
            Tuple of batched tensors

        EXAMPLE:
        >>> # batch = [(Tensor([1,2]), Tensor(0)),
        ...            (Tensor([3,4]), Tensor(1))]
        >>> # Returns: (Tensor([[1,2], [3,4]]), Tensor([0, 1]))
        """
        if len(batch) == 0:
            return ()

        # Determine number of tensors per sample
        num_tensor = len(batch[0])

        # Group tensors by position
        batched_tensors = []
        for idx in range(num_tensor):
            # Extract all tensors at this position
            tensor_list = [sample[idx].data for sample in batch]

            # Stack into batch tensor
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)
