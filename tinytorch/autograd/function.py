"""
Function base class for autograd operations.

Following PyTorch's pattern, each operation is a Function subclass
that implements both forward() and backward() static methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .context import Context

if TYPE_CHECKING:
    from ..tensor import Tensor


class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.
    """

    def __init__(self):
        self.ctx = None
        self.next_functions = []

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from ..tensor import Tensor

        # The forward pass code will work directly on the underlying data, not tensor
        raw_inputs = tuple(arg.data if isinstance(arg, Tensor) else arg for arg in args)
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )
        ctx = Context() if requires_grad else None
        output_data = cls.forward(ctx, *raw_inputs, **kwargs)
        output = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:
            # Create the Function node instance if we need to track gradient
            node = cls()
            node.ctx = ctx
            for arg in args:
                if isinstance(arg, Tensor) and arg.requires_grad:
                    if arg.grad_fn:
                        # The input already has Function node constructed.
                        node.next_functions.append(arg.grad_fn)
                    else:
                        # This is a leaf node that requires gradient, so we create
                        # AccumulateGrad object for it.
                        node.next_functions.append(arg.get_accumulate_grad())
                else:
                    node.next_functions.append(None)
            output.grad_fn = node

        return output


class AccumulateGrad(Function):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.variable = tensor

    def backward(self, grad_output):
        if self.variable.grad is None:
            self.variable.grad = grad_output
        else:
            self.variable.grad += grad_output


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Reduce gradient to match a broadcasted input's shape.
    """
    if grad.shape == target_shape:
        return grad
    g = grad

    # Remove leading broadcasted dims
    while len(g.shape) > len(target_shape):
        g = g.sum(axis=0)

    # Sum over axes where target dim was 1 (broadcasted)
    for i, dim in enumerate(target_shape):
        if dim == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class AddFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a_shape, b_shape = ctx.saved_inputs
        return _unbroadcast(grad_output, a_shape), _unbroadcast(grad_output, b_shape)


class SubFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a.shape, b.shape)
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a_shape, b_shape = ctx.saved_inputs
        return _unbroadcast(grad_output, a_shape), _unbroadcast(-grad_output, b_shape)


class MulFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_inputs
        # At this point, these match the 'broadcasted' shape
        grad_a = grad_output * b
        grad_b = grad_output * a
        # We sum across broadcasted dimensions back to original shapes
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class DivFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_inputs
        # d/da (a/b) = 1/b
        # d/db (a/b) = -a/b^2
        grad_a = grad_output / b
        grad_b = grad_output * (-a / (b**2))
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class NegFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)
        if ctx is not None:
            ctx.save_for_backward(a.shape)
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (a_shape,) = ctx.saved_inputs
        return (_unbroadcast(-grad_output, a_shape),)


class PowFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a, b)
        return a**b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_inputs
        grad_a = grad_output * b * a ** (b - 1)
        grad_b = grad_output * a**b * np.log(a)
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class ExpFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        result = np.exp(a)
        if ctx is not None:
            ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (result,) = ctx.saved_inputs
        grad_a = grad_output * result
        return (_unbroadcast(grad_a, result.shape),)


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, b) = args
        a, b = np.asanyarray(a), np.asanyarray(b)
        if ctx is not None:
            ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_inputs

        # Swap only the last two axes: (..., i, j) -> (..., j, i)
        # Using axes=None on a 2D array is fine, but for N-D, we need specific axes.
        def rel_transpose(x: np.ndarray) -> np.ndarray:
            return x.swapaxes(-1, -2) if x.ndim >= 2 else x.T

        grad_a = grad_output @ rel_transpose(b)
        grad_b = rel_transpose(a) @ grad_output

        # Note: If 'a' or 'b' were broadcasted (e.g., [1, 3, 4] @ [4, 5]),
        # we still need _unbroadcast here!
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, shape) = args
        a = np.asanyarray(a)
        if ctx is not None:
            # Save the ORIGINAL shape for backward
            ctx.save_for_backward(a.shape)
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (original_shape,) = ctx.saved_inputs
        # Reshape gradient back to original shape
        grad_a = grad_output.reshape(original_shape)
        return (grad_a,)


class TransposeFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args, axes=None, swap_mode=False) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)

        if swap_mode and axes is not None and len(axes) == 2:
            # Swap two axes using swapaxes
            if ctx is not None:
                ctx.save_for_backward(axes, True)  # True = swap_mode
            return a.swapaxes(axes[0], axes[1])
        else:
            # Full permutation using transpose
            if ctx is not None:
                ctx.save_for_backward(axes, False)  # False = full permutation
            return a.transpose(axes)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        axes, swap_mode = ctx.saved_inputs

        if swap_mode:
            # Swapping is its own inverse
            return (grad_output.swapaxes(axes[0], axes[1]),)

        if axes is None:
            # Default transpose reverses all dims; it is its own inverse
            return (grad_output.transpose(),)

        # Calculate inverse permutation
        # If axes is [1, 0, 2], inv_axes is also [1, 0, 2]
        # If axes is [2, 0, 1], inv_axes is [1, 2, 0]
        inv_axes = np.argsort(axes)
        return (grad_output.transpose(inv_axes),)


class SumFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args, axis=None, keepdims=False) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)
        if ctx is not None:
            ctx.save_for_backward(a.shape, axis, keepdims)
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        input_shape, axis, keepdims = ctx.saved_inputs

        # 1. If keepdims was False, the axes were collapsed.
        # We must add them back as dimensions of size 1 so broadcasting works.
        if not keepdims and axis is not None:
            # If axis was an integer, make it a list for uniform processing
            actual_axes = [axis] if isinstance(axis, int) else axis

            # Re-insert the reduced dimensions
            grad_reshape = list(grad_output.shape)
            for ax in sorted(actual_axes):
                # Handle negative indexing (e.g., axis=-1)
                pos = ax if ax >= 0 else len(input_shape) + ax
                grad_reshape.insert(pos, 1)

            grad_output = grad_output.reshape(grad_reshape)

        # 2. Now that shapes are aligned, NumPy broadcasting handles the "expansion"
        # grad_output (1, 10) broadcasted to input_shape (5, 10)
        grad_a = np.broadcast_to(grad_output, input_shape)

        return (grad_a,)


class MeanFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args, axis=None, keepdims=False) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)

        # 1. Perform the mean
        result = a.mean(axis=axis, keepdims=keepdims)

        # 2. Calculate n: how many elements were averaged?
        # a.size is total elements. result.size is output elements.
        # Dividing them gives the count of elements per "reduction group".
        n = a.size // result.size

        if ctx is not None:
            ctx.save_for_backward(a.shape, axis, keepdims, n)

        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        input_shape, axis, keepdims, n = ctx.saved_inputs

        # 1. Restore dimensions if keepdims was False (identical to Sum)
        if not keepdims and axis is not None:
            actual_axes = [axis] if isinstance(axis, int) else axis
            grad_reshape = list(grad_output.shape)
            for ax in sorted(actual_axes):
                pos = ax if ax >= 0 else len(input_shape) + ax
                grad_reshape.insert(pos, 1)
            grad_output = grad_output.reshape(grad_reshape)

        # 2. Broadcast and Scale
        # In Sum, we just broadcast. In Mean, we broadcast and divide by n.
        grad_a = np.broadcast_to(grad_output, input_shape) / n

        return (grad_a,)


class MaxFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args, axis=None, keepdims=False) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)

        # 1. Compute the maximum
        result = a.max(axis=axis, keepdims=keepdims)

        # 2. Create a mask of where the maximum occurred
        # We need to make sure 'result' is broadcastable back to 'a'
        # to create the mask accurately.
        if not keepdims and axis is not None:
            # Temporarily restore dims for comparison if they were dropped
            res_for_mask = np.expand_dims(result, axis)
        else:
            res_for_mask = result

        mask = a == res_for_mask

        if ctx is not None:
            ctx.save_for_backward(mask, a.shape, axis, keepdims)

        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        mask, input_shape, axis, keepdims = ctx.saved_inputs

        # 1. Restore dimensions of grad_output so it matches the mask's rank
        if not keepdims and axis is not None:
            actual_axes = [axis] if isinstance(axis, int) else axis
            grad_reshape = list(grad_output.shape)
            for ax in sorted(actual_axes):
                pos = ax if ax >= 0 else len(input_shape) + ax
                grad_reshape.insert(pos, 1)
            grad_output = grad_output.reshape(grad_reshape)

        # 2. Routing the gradient:
        # We multiply the incoming gradient by the mask.
        # Only the 'True' positions in the mask will let the gradient pass.
        grad_a = grad_output * mask

        # 3. Handling Ties (Optional but important)
        # If multiple elements were the max, we should technically
        # divide the gradient among them so the sum remains correct.
        # PyTorch often skips this, but it's cleaner for math:
        # counts = mask.sum(axis=axis, keepdims=True)
        # grad_a = grad_a / counts

        return (grad_a,)


# ============================================================================
# Activation Functions
# ============================================================================


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)
        if ctx is not None:
            ctx.save_for_backward(a)
        return np.maximum(0, a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (a,) = ctx.saved_inputs
        # Gradient is 1 where input > 0, else 0
        grad_a = grad_output * (a > 0).astype(grad_output.dtype)
        return (grad_a,)


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        z = np.asanyarray(a)
        # Numerically stable sigmoid using mask-based computation
        # This avoids computing exp(z) for large positive z (which overflows)
        result = np.zeros_like(z)
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        # For z >= 0: sigmoid(z) = 1 / (1 + exp(-z))
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        # For z < 0: sigmoid(z) = exp(z) / (1 + exp(z))
        exp_z = np.exp(z[neg_mask])
        result[neg_mask] = exp_z / (1 + exp_z)
        if ctx is not None:
            ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (result,) = ctx.saved_inputs
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        grad_a = grad_output * result * (1 - result)
        return (grad_a,)


class TanhFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)
        result = np.tanh(a)
        if ctx is not None:
            ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        (result,) = ctx.saved_inputs
        # d/dx tanh(x) = 1 - tanh^2(x)
        grad_a = grad_output * (1 - result**2)
        return (grad_a,)


class GELUFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a,) = args
        a = np.asanyarray(a)
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_over_pi * (a + 0.044715 * a**3)
        tanh_inner = np.tanh(inner)
        result = 0.5 * a * (1 + tanh_inner)
        if ctx is not None:
            ctx.save_for_backward(a, tanh_inner)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        a, tanh_inner = ctx.saved_inputs
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        sech2 = 1 - tanh_inner**2  # sech^2(x) = 1 - tanh^2(x)
        inner_deriv = sqrt_2_over_pi * (1 + 3 * 0.044715 * a**2)
        # d/dx [0.5 * x * (1 + tanh(f(x)))]
        # = 0.5 * (1 + tanh(f(x))) + 0.5 * x * sech^2(f(x)) * f'(x)
        grad_a = grad_output * (0.5 * (1 + tanh_inner) + 0.5 * a * sech2 * inner_deriv)
        return (grad_a,)


class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (a, axis) = args
        a = np.asanyarray(a)
        # Numerically stable softmax: subtract max before exp
        a_max = a.max(axis=axis, keepdims=True)
        exp_a = np.exp(a - a_max)
        result = exp_a / exp_a.sum(axis=axis, keepdims=True)
        if ctx is not None:
            ctx.save_for_backward(result, axis)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        result, axis = ctx.saved_inputs
        # Softmax backward: grad_input = y * (grad_output - sum(grad_output * y, axis))
        sum_term = (grad_output * result).sum(axis=axis, keepdims=True)
        grad_a = result * (grad_output - sum_term)
        return (grad_a,)


# ============================================================================
# Loss Functions
# ============================================================================


class MSEFunction(Function):
    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (pred, target) = args
        pred, target = np.asanyarray(pred), np.asanyarray(target)
        diff = pred - target
        result = np.mean(diff**2)
        if ctx is not None:
            ctx.save_for_backward(diff, pred.size)
        return np.array(result)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        diff, n = ctx.saved_inputs
        # d/dpred MSE = 2 * (pred - target) / n
        grad_pred = grad_output * 2 * diff / n
        # d/dtarget MSE = -2 * (pred - target) / n
        grad_target = -grad_pred
        return grad_pred, grad_target


class CrossEntropyFunction(Function):
    """
    Cross-entropy loss for multi-class classification.
    Takes logits (unnormalized scores) and target class indices.
    Combines softmax and negative log-likelihood for numerical stability.
    """

    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (logits, targets) = args
        logits = np.asanyarray(logits)
        targets = np.asanyarray(targets).astype(np.int64)

        # Compute softmax (numerically stable)
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Compute cross-entropy loss: -log(prob of correct class)
        batch_size = logits.shape[0]
        eps = 1e-12
        log_probs = np.log(probs + eps)
        loss = -log_probs[np.arange(batch_size), targets].mean()

        if ctx is not None:
            ctx.save_for_backward(probs, targets, batch_size)

        return np.array(loss)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        probs, targets, batch_size = ctx.saved_inputs

        # Gradient of cross-entropy with softmax: probs - one_hot(targets)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1
        grad_logits = grad_output * grad_logits / batch_size

        # No gradient for targets (class indices are not differentiable)
        return (grad_logits, None)


class BinaryCrossEntropyFunction(Function):
    """
    Binary cross-entropy loss for binary classification.
    Takes predictions (after sigmoid) and binary targets (0 or 1).
    """

    @staticmethod
    def forward(ctx: Context, *args) -> np.ndarray:
        (pred, target) = args
        pred, target = np.asanyarray(pred), np.asanyarray(target)

        # Clip predictions for numerical stability
        # Use 1e-7 as eps since float32 has ~7 digits of precision
        eps = 1e-7
        pred_clipped = np.clip(pred, eps, 1 - eps)

        # BCE: -mean(target * log(pred) + (1 - target) * log(1 - pred))
        loss = -np.mean(
            target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped)
        )

        if ctx is not None:
            ctx.save_for_backward(pred_clipped, target, pred.size)

        return np.array(loss)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        pred, target, n = ctx.saved_inputs

        # d/dpred BCE = (-target/pred + (1-target)/(1-pred)) / n
        grad_pred = grad_output * (-target / pred + (1 - target) / (1 - pred)) / n

        # d/dtarget BCE = (-log(pred) + log(1-pred)) / n
        # Usually targets are not trainable, but included for completeness
        grad_target = grad_output * (-np.log(pred) + np.log(1 - pred)) / n

        return (grad_pred, grad_target)
