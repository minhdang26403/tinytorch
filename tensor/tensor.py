import numpy as np

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class Tensor:
    """

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        """String representation of tensor for debugging."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes.

        Systems Concept: Understanding memory footprint is fundamental to ML systems.
        Before running any operation, engineers should know how much memory it requires.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """Subtract two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """Multiply two tensors element-wise (NOT matrix multiplication)."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """Divide two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """Matrix multiplication of two tensors."""

        if not isinstance(other, Tensor):
            raise TypeError(
                f"Expected Tensor for matrix multiplication, got {type(other)}"
            )

        return Tensor(self.data @ other.data)

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors."""
        return Tensor(self.data[key])

    def reshape(self, *shape):
        """Reshape tensor to new dimensions."""
        # If the user passed a single tuple, e.g., reshape((1, 2, 3))
        # 'shape' will be ((1, 2, 3),)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = shape[0]
        else:
            new_shape = shape

        return Tensor(self.data.reshape(new_shape))

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis."""
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis."""
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis."""
        return Tensor(self.data.max(axis=axis, keepdims=keepdims))
