from abc import ABC, abstractmethod

from ..tensor import Tensor


class Dataset(ABC):
    """
    Abstract base class for all datasets.

    Provides the fundamental interface that all datasets must implement:
    - __len__(): Returns the total number of samples
    - __getitem__(idx): Returns the sample at given index
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        This method must be implemented by all subclasses to enable
        len(dataset) calls and batch size calculations.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Return the sample at the given index.

        Args:
            idx: Index of the sample to retrieve (0 <= idx < len(dataset))

        Returns:
            The sample at index idx. Format depends on the dataset implementation.
            Could be (data, label) tuple, single tensor, etc.
        """
        pass


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors for supervised learning.

    Each sample is a tuple of tensors from the same index across all input tensors.
    All tensors must have the same size in their first dimension.

    EXAMPLE:
    >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features each
    >>> labels = Tensor([0, 1, 0])                    # 3 labels
    >>> dataset = TensorDataset(features, labels)
    >>> print(len(dataset))  # 3
    >>> print(dataset[1])    # (Tensor([3, 4]), Tensor(1))
    """

    def __init__(self, *tensors: Tensor):
        assert len(tensors) > 0, "Must provide at least one tensor"

        # Validate all tensors have same first dimension
        first_size = tensors[0].shape[0]  # Size of first dimension
        for i, tensor in enumerate(tensors):
            if tensor.shape[0] != first_size:
                raise ValueError(
                    f"All tensors must have same size in first dimension. "
                    f"Tensor 0: {first_size}, Tensor {i}: {tensor.shape[0]}"
                )

        self.tensors = tensors

    def __len__(self):
        """
        Return number of samples (size of first dimension).

        EXAMPLE:
        >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples
        >>> labels = Tensor([0, 1, 0])
        >>> dataset = TensorDataset(features, labels)
        >>> print(len(dataset))  # 3
        """
        return self.tensors[0].shape[0]

    def __getitem__(self, idx: int):
        """
        Return tuple of tensor slices at given index.

        Args:
            idx: Sample index

        Returns:
            Tuple containing tensor[idx] for each input tensor

        EXAMPLE:
        >>> features = Tensor([[1, 2], [3, 4], [5, 6]])
        >>> labels = Tensor([0, 1, 0])
        >>> dataset = TensorDataset(features, labels)
        >>> sample = dataset[1]
        >>> # Returns: (Tensor([3, 4]), Tensor(1))
        """
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        # Return tuple of slices from all tensors
        return tuple(tensor[idx] for tensor in self.tensors)
