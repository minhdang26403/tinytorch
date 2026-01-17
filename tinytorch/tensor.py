from __future__ import annotations

import numpy as np

from tinytorch.autograd import (
    AccumulateGrad,
    AddFunction,
    DivFunction,
    ExpFunction,
    Function,
    MatmulFunction,
    MaxFunction,
    MeanFunction,
    MulFunction,
    PowFunction,
    ReshapeFunction,
    SubFunction,
    SumFunction,
    TransposeFunction,
)


class Tensor:
    """

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data, *, requires_grad=False):
        # Internal storage of the tensor
        self.data = np.array(data, dtype=np.float32)

        # Metadata about the tensor
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

        # Internal state for gradient tracking
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None

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

    @property
    def grad_fn(self):
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, value):
        self._grad_fn = value

    def get_accumulate_grad(self):
        if not self.requires_grad:
            return None

        if self._grad_fn is not None:
            return self._grad_fn

        # If it's a leaf and needs grad, create the one and only AccumulateGrad node
        self._grad_fn = AccumulateGrad(self)
        return self._grad_fn

    def __getitem__(self, key):
        """
        Enable indexing and slicing operations on Tensors.
        """
        return Tensor(self.data[key], requires_grad=self.requires_grad)

    def __add__(self, other):
        """
        Add two tensors element-wise with broadcasting support.
        """
        return AddFunction.apply(self, other)

    def __radd__(self, other):
        return Tensor(other).__add__(self)

    def __sub__(self, other):
        """
        Subtract two tensors element-wise.
        """

        return SubFunction.apply(self, other)

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __neg__(self):
        """
        Negate the tensor.
        """

        return SubFunction.apply(0, self)

    def __mul__(self, other):
        """
        Multiply two tensors element-wise (NOT matrix multiplication).
        """
        return MulFunction.apply(self, other)

    def __rmul__(self, other):
        return Tensor(other).__mul__(self)

    def __truediv__(self, other):
        """
        Divide two tensors element-wise.
        """
        return DivFunction.apply(self, other)

    def __pow__(self, other):
        """
        Raise the tensor to the power of another tensor.
        """

        return PowFunction.apply(self, other)

    def exp(self):
        """
        Compute exponential of tensor.
        """
        return ExpFunction.apply(self)

    def matmul(self, other):
        """
        Matrix multiplication of two tensors.
        """
        return MatmulFunction.apply(self, other)

    def __matmul__(self, other):
        """
        Enable @ operator for matrix multiplication.
        """
        return self.matmul(other)

    def reshape(self, *shape):
        """
        Reshape tensor to new dimensions with basic validation.
        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        return ReshapeFunction.apply(self, shape=shape)

    def transpose(self, *args, axes=None):
        """
        Transpose tensor dimensions.

        Can be called as:
        - transpose(): reverse all dimensions
        - transpose(axes=(2, 0, 1)): full permutation
        - transpose(0, 2) or transpose(axes=(0, 2)): swap two axes
        """
        # Handle positional args: transpose(0, 2) means swap axes 0 and 2
        if args:
            axes = args

        # If axes is a 2-tuple on an N-D array (N > 2), treat as swapaxes
        if axes is not None and len(axes) == 2 and len(self.shape) > 2:
            return TransposeFunction.apply(self, axes=axes, swap_mode=True)

        return TransposeFunction.apply(self, axes=axes, swap_mode=False)

    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor along specified axis.
        """
        return SumFunction.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """
        Compute mean of tensor along specified axis.
        """

        return MeanFunction.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """
        Find maximum values along specified axis.
        """
        return MaxFunction.apply(self, axis=axis, keepdims=keepdims)

    def backward(self, gradient: "Tensor" | None = None) -> None:
        """
        Backpropagate gradients through the computation graph.
        """

        if not self.requires_grad or not self._grad_fn:
            raise ValueError(
                "The tensor does not require grad and does not have a grad_fn"
            )

        # Topological sort of Function nodes
        topological_order = Tensor._find_topological_order(self.grad_fn)

        # Initialize the gradient flow
        # We use a dictionary to store gradients for each function node
        grads = {}
        if gradient is None:
            grads[self.grad_fn] = np.ones_like(self.data)
        else:
            grads[self.grad_fn] = gradient.data

        for node in reversed(topological_order):
            # Get the gradient of this node
            grad_output = grads.pop(node)
            # Use the output gradient to compute the gradient of this node's inputs.
            # If it's a math op, it returns gradients for inputs.
            # If it's AccumulateGrad, it writes to .grad and returns None.
            if isinstance(node, AccumulateGrad):
                node.backward(grad_output)
                continue

            grad_inputs = node.backward(node.ctx, grad_output)

            for grad_input, next_node in zip(grad_inputs, node.next_functions):
                if next_node is not None:
                    if next_node not in grads:
                        grads[next_node] = grad_input
                    else:
                        grads[next_node] += grad_input

    @staticmethod
    def _find_topological_order(node: Function) -> list[Function]:
        visited = set()
        topological_order = []

        def dfs(node: Function):
            if node in visited:
                return

            visited.add(node)
            for next_func in node.next_functions:
                if next_func is not None:
                    dfs(next_func)

            topological_order.append(node)

        dfs(node)

        return topological_order
