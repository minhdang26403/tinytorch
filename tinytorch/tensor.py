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

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        """
        String representation of tensor for debugging.
        """
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """
        Human-readable string representation.
        """
        return f"Tensor({self.data})"

    def numpy(self):
        """
        Return the underlying NumPy array.
        """
        return self.data

    def memory_footprint(self):
        """
        Calculate exact memory usage in bytes.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    def __add__(self, other):
        """
        Add two tensors element-wise with broadcasting support.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """
        Subtract two tensors element-wise.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """
        Multiply two tensors element-wise (NOT matrix multiplication).
        """
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """
        Divide two tensors element-wise.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """
        Matrix multiplication of two tensors.
        """

        if not isinstance(other, Tensor):
            raise TypeError(
                f"Expected Tensor for matrix multiplication, got {type(other)}"
            )

        # Validate inner dimensions before dispatching to NumPy so we can
        # provide a clearer error message (helpful in tests and debugging).
        self_inner = self.data.shape[-1]
        # For vectors, the relevant dimension is the only axis (index 0); for
        # matrices/higher rank tensors, it's the second-to-last axis.
        other_inner = (
            other.data.shape[-2] if other.data.ndim > 1 else other.data.shape[0]
        )

        if self_inner != other_inner:
            raise ValueError(
                f"Inner dimensions must match for matmul: {self_inner} ≠ {other_inner}"
            )

        return Tensor(self.data @ other.data)

    def __matmul__(self, other):
        """
        Enable @ operator for matrix multiplication.
        """
        return self.matmul(other)

    def __getitem__(self, key):
        """
        Enable indexing and slicing operations on Tensors.
        """
        return Tensor(self.data[key])

    def reshape(self, *shape):
        """
        Reshape tensor to new dimensions with basic validation.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        # Handle a single inferred dimension (-1) manually for clearer errors.
        if shape.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred with -1")

        if -1 in shape:
            known = 1
            for dim in shape:
                if dim != -1:
                    known *= dim
            if self.size % known != 0:
                raise ValueError(
                    f"Total elements must match for reshape: {self.size} ≠ {known}"
                )
            inferred = self.size // known
            computed_shape = tuple(inferred if dim == -1 else dim for dim in shape)
        else:
            requested = 1
            for dim in shape:
                requested *= dim
            if requested != self.size:
                raise ValueError(
                    f"Total elements must match for reshape: {self.size} ≠ {requested}"
                )
            computed_shape = shape

        return Tensor(self.data.reshape(computed_shape))

    def transpose(self, dim0=None, dim1=None):
        """
        Transpose tensor dimensions.
        """
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
        """
        Sum tensor along specified axis.
        """
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """
        Compute mean of tensor along specified axis.
        """
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        """
        Find maximum values along specified axis.
        """
        return Tensor(self.data.max(axis=axis, keepdims=keepdims))
