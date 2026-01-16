"""
Unit tests for autograd functions.

Tests both forward pass correctness and backward pass gradients using
numerical gradient checking.
"""

import numpy as np
import pytest

from tinytorch import Tensor

# =============================================================================
# Helper Functions
# =============================================================================


def numerical_gradient(f, x, eps=1e-4):
    """
    Compute numerical gradient using central difference.

    Args:
        f: Function that takes a Tensor and returns a scalar Tensor
        x: Input Tensor
        eps: Small perturbation for finite difference

    Returns:
        Numerical gradient as numpy array
    """
    # Use float64 for more precise numerical gradient
    grad = np.zeros(x.data.shape, dtype=np.float64)
    original_data = x.data.copy().astype(np.float64)

    it = np.nditer(original_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = original_data[idx]

        # f(x + eps)
        x.data = original_data.copy().astype(np.float32)
        x.data[idx] = float(old_val + eps)
        fxp = float(f(x).data)

        # f(x - eps)
        x.data = original_data.copy().astype(np.float32)
        x.data[idx] = float(old_val - eps)
        fxm = float(f(x).data)

        # Central difference
        grad[idx] = (fxp - fxm) / (2 * eps)

        it.iternext()

    # Restore original data
    x.data = original_data.astype(np.float32)
    return grad.astype(np.float32)


def check_gradient(f, x, atol=5e-3, rtol=5e-2):
    """
    Check that analytical gradient matches numerical gradient.

    Args:
        f: Function that takes a Tensor and returns a scalar Tensor
        x: Input Tensor with requires_grad=True
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if gradients match, raises AssertionError otherwise
    """
    # Compute analytical gradient
    x.grad = None
    y = f(x)
    y.backward()
    analytical = x.grad.copy()

    # Compute numerical gradient
    numerical = numerical_gradient(f, x)

    # Compare
    np.testing.assert_allclose(
        analytical,
        numerical,
        atol=atol,
        rtol=rtol,
        err_msg=f"Gradient mismatch!\nAnalytical:\n{analytical}\n"
        f"Numerical:\n{numerical}",
    )
    return True


# =============================================================================
# Arithmetic Operations Tests
# =============================================================================


class TestAddFunction:
    def test_forward_basic(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b
        np.testing.assert_array_equal(c.data, [5.0, 7.0, 9.0])

    def test_forward_broadcasting(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 20.0])
        c = a + b
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_array_equal(c.data, expected)

    def test_forward_scalar(self):
        a = Tensor([1.0, 2.0, 3.0])
        c = a + 10
        np.testing.assert_array_equal(c.data, [11.0, 12.0, 13.0])

    def test_backward_basic(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a + b
        c.sum().backward()
        np.testing.assert_array_equal(a.grad, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [1.0, 1.0, 1.0])

    def test_backward_broadcasting(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([10.0, 20.0], requires_grad=True)
        c = a + b
        c.sum().backward()
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0], [1.0, 1.0]])
        # b is broadcasted, so gradient is summed over broadcast dimension
        np.testing.assert_array_equal(b.grad, [2.0, 2.0])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4))
        check_gradient(lambda t: (t + y).sum(), x)


class TestSubFunction:
    def test_forward_basic(self):
        a = Tensor([5.0, 7.0, 9.0])
        b = Tensor([1.0, 2.0, 3.0])
        c = a - b
        np.testing.assert_array_equal(c.data, [4.0, 5.0, 6.0])

    def test_backward_basic(self):
        a = Tensor([5.0, 7.0, 9.0], requires_grad=True)
        b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a - b
        c.sum().backward()
        np.testing.assert_array_equal(a.grad, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [-1.0, -1.0, -1.0])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4))
        check_gradient(lambda t: (t - y).sum(), x)


class TestMulFunction:
    def test_forward_basic(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a * b
        np.testing.assert_array_equal(c.data, [4.0, 10.0, 18.0])

    def test_backward_basic(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a * b
        c.sum().backward()
        # d/da (a*b) = b
        np.testing.assert_array_equal(a.grad, [4.0, 5.0, 6.0])
        # d/db (a*b) = a
        np.testing.assert_array_equal(b.grad, [1.0, 2.0, 3.0])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4))
        check_gradient(lambda t: (t * y).sum(), x)


class TestDivFunction:
    def test_forward_basic(self):
        a = Tensor([4.0, 10.0, 18.0])
        b = Tensor([2.0, 5.0, 6.0])
        c = a / b
        np.testing.assert_array_equal(c.data, [2.0, 2.0, 3.0])

    def test_backward_basic(self):
        a = Tensor([4.0, 10.0, 18.0], requires_grad=True)
        b = Tensor([2.0, 5.0, 6.0], requires_grad=True)
        c = a / b
        c.sum().backward()
        # d/da (a/b) = 1/b
        np.testing.assert_allclose(a.grad, [0.5, 0.2, 1 / 6])
        # d/db (a/b) = -a/b^2
        np.testing.assert_allclose(b.grad, [-1.0, -0.4, -0.5])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4) + 2)  # Avoid division by zero
        check_gradient(lambda t: (t / y).sum(), x)


class TestPowFunction:
    def test_forward_basic(self):
        a = Tensor([2.0, 3.0, 4.0])
        c = a**2
        np.testing.assert_array_equal(c.data, [4.0, 9.0, 16.0])

    def test_backward_basic(self):
        a = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        c = a**2
        c.sum().backward()
        # d/da (a^2) = 2a
        np.testing.assert_array_equal(a.grad, [4.0, 6.0, 8.0])

    def test_gradient_check(self):
        x = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        check_gradient(lambda t: (t**2.5).sum(), x)


class TestExpFunction:
    def test_forward_basic(self):
        a = Tensor([0.0, 1.0, 2.0])
        c = a.exp()
        np.testing.assert_allclose(c.data, [1.0, np.e, np.e**2])

    def test_backward_basic(self):
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        c = a.exp()
        c.sum().backward()
        # d/da exp(a) = exp(a)
        np.testing.assert_allclose(a.grad, [1.0, np.e, np.e**2])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)  # Small values
        check_gradient(lambda t: t.exp().sum(), x)


# =============================================================================
# Matrix Operations Tests
# =============================================================================


class TestMatmulFunction:
    def test_forward_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(c.data, expected)

    def test_backward_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = a @ b
        c.sum().backward()
        # d/da (a @ b) = grad @ b.T
        # d/db (a @ b) = a.T @ grad
        expected_a_grad = np.array([[11.0, 15.0], [11.0, 15.0]])
        expected_b_grad = np.array([[4.0, 4.0], [6.0, 6.0]])
        np.testing.assert_array_equal(a.grad, expected_a_grad)
        np.testing.assert_array_equal(b.grad, expected_b_grad)

    def test_gradient_check(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5))
        check_gradient(lambda t: (t @ b).sum(), a)


# =============================================================================
# Shape Operations Tests
# =============================================================================


class TestReshapeFunction:
    def test_forward_basic(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = a.reshape(2, 3)
        assert b.shape == (2, 3)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(b.data, expected)

    def test_backward_basic(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        b = a.reshape(2, 3)
        (b * 2).sum().backward()
        np.testing.assert_array_equal(a.grad, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


class TestTransposeFunction:
    def test_forward_2d(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.transpose()
        assert b.shape == (3, 2)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(b.data, expected)

    def test_backward_2d(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = a.transpose()
        (b * 2).sum().backward()
        np.testing.assert_array_equal(a.grad, [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

    def test_swap_axes_3d(self):
        a = Tensor(
            np.arange(24).reshape(2, 3, 4).astype(np.float32), requires_grad=True
        )
        b = a.transpose(0, 2)  # Swap axes 0 and 2
        assert b.shape == (4, 3, 2)
        b.sum().backward()
        assert a.grad.shape == (2, 3, 4)


# =============================================================================
# Reduction Operations Tests
# =============================================================================


class TestSumFunction:
    def test_forward_all(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum()
        assert b.data == 10.0

    def test_forward_axis(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum(axis=0)
        np.testing.assert_array_equal(b.data, [4.0, 6.0])
        c = a.sum(axis=1)
        np.testing.assert_array_equal(c.data, [3.0, 7.0])

    def test_backward_all(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.sum()
        b.backward()
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0], [1.0, 1.0]])

    def test_backward_axis(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.sum(axis=0)
        b.sum().backward()
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0], [1.0, 1.0]])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradient(lambda t: t.sum(), x)
        check_gradient(lambda t: t.sum(axis=0).sum(), x)
        check_gradient(lambda t: t.sum(axis=1).sum(), x)


class TestMeanFunction:
    def test_forward_all(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.mean()
        assert b.data == 2.5

    def test_forward_axis(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.mean(axis=0)
        np.testing.assert_array_equal(b.data, [2.0, 3.0])

    def test_backward_all(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.mean()
        b.backward()
        np.testing.assert_array_equal(a.grad, [[0.25, 0.25], [0.25, 0.25]])

    def test_gradient_check(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradient(lambda t: t.mean(), x)
        check_gradient(lambda t: t.mean(axis=0).sum(), x)


class TestMaxFunction:
    def test_forward_all(self):
        a = Tensor([[1.0, 5.0], [3.0, 4.0]])
        b = a.max()
        assert b.data == 5.0

    def test_forward_axis(self):
        a = Tensor([[1.0, 5.0], [3.0, 4.0]])
        b = a.max(axis=0)
        np.testing.assert_array_equal(b.data, [3.0, 5.0])

    def test_backward_basic(self):
        a = Tensor([[1.0, 5.0], [3.0, 4.0]], requires_grad=True)
        b = a.max()
        b.backward()
        # Gradient flows only to the max element
        np.testing.assert_array_equal(a.grad, [[0.0, 1.0], [0.0, 0.0]])


# =============================================================================
# Activation Functions Tests
# =============================================================================


class TestReLUFunction:
    def test_forward_basic(self):
        from tinytorch.nn import ReLU

        relu = ReLU()
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = relu(a)
        np.testing.assert_array_equal(b.data, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_backward_basic(self):
        from tinytorch.nn import ReLU

        relu = ReLU()
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        b = relu(a)
        b.sum().backward()
        # Gradient is 1 where input > 0, else 0
        np.testing.assert_array_equal(a.grad, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_gradient_check(self):
        from tinytorch.nn import ReLU

        relu = ReLU()
        # Avoid values near 0 where gradient is undefined
        x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]), requires_grad=True)
        check_gradient(lambda t: relu(t).sum(), x)


class TestSigmoidFunction:
    def test_forward_basic(self):
        from tinytorch.nn import Sigmoid

        sigmoid = Sigmoid()
        a = Tensor([0.0])
        b = sigmoid(a)
        np.testing.assert_allclose(b.data, [0.5])

    def test_forward_extreme(self):
        from tinytorch.nn import Sigmoid

        sigmoid = Sigmoid()
        a = Tensor([-1000.0, 1000.0])
        b = sigmoid(a)
        np.testing.assert_allclose(b.data, [0.0, 1.0], atol=1e-6)

    def test_backward_basic(self):
        from tinytorch.nn import Sigmoid

        sigmoid = Sigmoid()
        a = Tensor([0.0], requires_grad=True)
        b = sigmoid(a)
        b.sum().backward()
        # d/dx sigmoid(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        np.testing.assert_allclose(a.grad, [0.25])

    def test_gradient_check(self):
        from tinytorch.nn import Sigmoid

        sigmoid = Sigmoid()
        x = Tensor(np.random.randn(3, 4) * 2, requires_grad=True)
        check_gradient(lambda t: sigmoid(t).sum(), x)


class TestTanhFunction:
    def test_forward_basic(self):
        from tinytorch.nn import Tanh

        tanh = Tanh()
        a = Tensor([0.0])
        b = tanh(a)
        np.testing.assert_allclose(b.data, [0.0])

    def test_backward_basic(self):
        from tinytorch.nn import Tanh

        tanh = Tanh()
        a = Tensor([0.0], requires_grad=True)
        b = tanh(a)
        b.sum().backward()
        # d/dx tanh(0) = 1 - tanh^2(0) = 1 - 0 = 1
        np.testing.assert_allclose(a.grad, [1.0])

    def test_gradient_check(self):
        from tinytorch.nn import Tanh

        tanh = Tanh()
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradient(lambda t: tanh(t).sum(), x)


class TestGELUFunction:
    def test_forward_basic(self):
        from tinytorch.nn import GELU

        gelu = GELU()
        a = Tensor([0.0])
        b = gelu(a)
        np.testing.assert_allclose(b.data, [0.0], atol=1e-6)

    def test_gradient_check(self):
        from tinytorch.nn import GELU

        gelu = GELU()
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradient(lambda t: gelu(t).sum(), x)


class TestSoftmaxFunction:
    def test_forward_basic(self):
        from tinytorch.nn import Softmax

        softmax = Softmax()
        a = Tensor([1.0, 2.0, 3.0])
        b = softmax(a)
        # Should sum to 1
        np.testing.assert_allclose(b.data.sum(), 1.0)
        # Largest input should have largest output
        assert np.argmax(b.data) == 2

    def test_forward_2d(self):
        from tinytorch.nn import Softmax

        softmax = Softmax(dim=-1)
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = softmax(a)
        # Each row should sum to 1
        np.testing.assert_allclose(b.data.sum(axis=-1), [1.0, 1.0])

    def test_gradient_check(self):
        from tinytorch.nn import Softmax

        softmax = Softmax()
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradient(lambda t: softmax(t).sum(), x)


# =============================================================================
# Loss Functions Tests
# =============================================================================


class TestMSEFunction:
    def test_forward_basic(self):
        from tinytorch.nn import MSELoss

        mse = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        loss = mse(pred, target)
        assert loss.data == 0.0

    def test_forward_nonzero(self):
        from tinytorch.nn import MSELoss

        mse = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([2.0, 3.0, 4.0])
        loss = mse(pred, target)
        # MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1 + 1 + 1) = 1
        assert loss.data == 1.0

    def test_backward_basic(self):
        from tinytorch.nn import MSELoss

        mse = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([2.0, 3.0, 4.0])
        loss = mse(pred, target)
        loss.backward()
        # d/dpred MSE = 2 * (pred - target) / n = 2 * [-1, -1, -1] / 3
        np.testing.assert_allclose(pred.grad, [-2 / 3, -2 / 3, -2 / 3])

    def test_gradient_check(self):
        from tinytorch.nn import MSELoss

        mse = MSELoss()
        pred = Tensor(np.random.randn(3, 4), requires_grad=True)
        target = Tensor(np.random.randn(3, 4))
        check_gradient(lambda t: mse(t, target), pred)


class TestCrossEntropyFunction:
    def test_forward_basic(self):
        from tinytorch.nn import CrossEntropyLoss

        ce = CrossEntropyLoss()
        # Perfect prediction for class 2
        logits = Tensor([[0.0, 0.0, 100.0]])
        targets = Tensor([2])
        loss = ce(logits, targets)
        np.testing.assert_allclose(loss.data, 0.0, atol=1e-5)

    def test_backward_basic(self):
        from tinytorch.nn import CrossEntropyLoss

        ce = CrossEntropyLoss()
        logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = Tensor([2])
        loss = ce(logits, targets)
        loss.backward()
        # Gradient should be softmax(logits) - one_hot(targets)
        assert logits.grad is not None
        assert logits.grad.shape == (1, 3)

    def test_gradient_check(self):
        from tinytorch.nn import CrossEntropyLoss

        ce = CrossEntropyLoss()
        logits = Tensor(np.random.randn(4, 5), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 3]))
        check_gradient(lambda t: ce(t, targets), logits)


class TestBinaryCrossEntropyFunction:
    def test_forward_perfect(self):
        from tinytorch.nn import BinaryCrossEntropyLoss

        bce = BinaryCrossEntropyLoss()
        pred = Tensor([0.999, 0.001])
        target = Tensor([1.0, 0.0])
        loss = bce(pred, target)
        assert loss.data < 0.01

    def test_backward_basic(self):
        from tinytorch.nn import BinaryCrossEntropyLoss

        bce = BinaryCrossEntropyLoss()
        pred = Tensor([0.5, 0.5], requires_grad=True)
        target = Tensor([1.0, 0.0])
        loss = bce(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_gradient_check(self):
        from tinytorch.nn import BinaryCrossEntropyLoss

        bce = BinaryCrossEntropyLoss()
        # Keep predictions away from 0 and 1 for stable gradients
        pred = Tensor(np.random.rand(4) * 0.8 + 0.1, requires_grad=True)
        target = Tensor(np.array([1.0, 0.0, 1.0, 0.0]))
        check_gradient(lambda t: bce(t, target), pred)


# =============================================================================
# Integration Tests
# =============================================================================


class TestComputationGraph:
    def test_chain_rule(self):
        """Test that gradients flow correctly through a chain of operations."""
        x = Tensor([2.0], requires_grad=True)
        y = x * 3  # y = 3x
        z = y + 1  # z = 3x + 1
        w = z**2  # w = (3x + 1)^2

        w.backward()
        # dw/dx = 2(3x + 1) * 3 = 6(3x + 1) = 6(7) = 42
        np.testing.assert_allclose(x.grad, [42.0])

    def test_multiple_paths(self):
        """Test gradient accumulation when a variable is used multiple times."""
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2
        z = x + y  # z = x + x^2

        z.backward()
        # dz/dx = 1 + 2x = 1 + 4 = 5
        np.testing.assert_allclose(x.grad, [5.0])

    def test_neural_network_forward_backward(self):
        """Test a simple neural network forward and backward pass."""
        # Input
        x = Tensor([[1.0, 2.0]], requires_grad=True)

        # Weights
        w1 = Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
        b1 = Tensor([0.1, 0.1], requires_grad=True)

        # Forward pass
        h = x @ w1 + b1  # Linear layer
        from tinytorch.nn import ReLU

        relu = ReLU()
        h = relu(h)  # Activation

        # Loss
        loss = h.sum()

        # Backward pass
        loss.backward()

        # Check that all gradients are computed
        assert x.grad is not None
        assert w1.grad is not None
        assert b1.grad is not None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
