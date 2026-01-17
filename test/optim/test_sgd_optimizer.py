import numpy as np

from tinytorch.optim import SGD
from tinytorch.tensor import Tensor


def test_unit_sgd_optimizer():
    """🔬 Test SGD optimizer implementation."""
    print("🔬 Unit Test: SGD Optimizer...")

    # Test basic SGD without momentum
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = SGD([param], lr=0.1)
    original_data = param.data.copy()

    optimizer.step()

    # Expected: param = param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    expected = original_data - 0.1 * param.grad.data
    assert np.allclose(param.data, expected)
    assert optimizer.step_count == 1

    # Test SGD with momentum
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])

    optimizer_momentum = SGD([param2], lr=0.1, momentum=0.9)

    # First step: v = 0.9 * 0 + [0.1, 0.2] = [0.1, 0.2]
    optimizer_momentum.step()
    expected_first = np.array([1.0, 2.0]) - 0.1 * np.array([0.1, 0.2])
    assert np.allclose(param2.data, expected_first)

    # Second step with same gradient
    param2.grad = Tensor([0.1, 0.2])
    optimizer_momentum.step()
    # v = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
    expected_momentum = np.array([0.19, 0.38])
    expected_second = expected_first - 0.1 * expected_momentum
    assert np.allclose(param2.data, expected_second, rtol=1e-5)

    # Test weight decay
    param3 = Tensor([1.0, 2.0], requires_grad=True)
    param3.grad = Tensor([0.1, 0.2])

    optimizer_wd = SGD([param3], lr=0.1, weight_decay=0.01)
    optimizer_wd.step()

    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    expected_wd = np.array([1.0, 2.0]) - 0.1 * np.array([0.11, 0.22])
    assert np.allclose(param3.data, expected_wd)

    print("✅ SGD optimizer works correctly!")
