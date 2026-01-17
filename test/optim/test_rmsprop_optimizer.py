import numpy as np

from tinytorch.optim import RMSProp
from tinytorch.tensor import Tensor


def test_unit_rmsprop_optimizer():
    """🔬 Test RMSProp optimizer implementation."""
    print("🔬 Unit Test: RMSProp Optimizer...")

    # Test basic RMSProp functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = RMSProp([param], lr=0.01, alpha=0.99, eps=1e-8)
    original_data = param.data.copy()

    # First step
    optimizer.step()

    # Manually compute expected values
    grad = np.array([0.1, 0.2])

    # Running average of squared gradients: v = 0.99 * 0 + 0.01 * grad^2
    v = (1 - 0.99) * (grad**2)

    # Update: param = param - lr * grad / (sqrt(v) + eps)
    expected = original_data - 0.01 * grad / (np.sqrt(v) + 1e-8)

    assert np.allclose(param.data, expected, rtol=1e-6)
    assert optimizer.step_count == 1

    # Test second step to verify running average accumulation
    param.grad = Tensor([0.1, 0.2])
    prev_data = param.data.copy()
    optimizer.step()

    # v should now be: 0.99 * v_prev + 0.01 * grad^2
    v_new = 0.99 * v + 0.01 * (grad**2)
    expected_second = prev_data - 0.01 * grad / (np.sqrt(v_new) + 1e-8)

    assert np.allclose(param.data, expected_second, rtol=1e-6)
    assert optimizer.step_count == 2
    assert optimizer.v_buffers[0] is not None

    # Test with weight decay
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])

    optimizer_wd = RMSProp([param2], lr=0.01, alpha=0.99, weight_decay=0.01)
    original_data2 = param2.data.copy()
    optimizer_wd.step()

    # Weight decay modifies the effective gradient
    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    grad_wd = np.array([0.1, 0.2]) + 0.01 * np.array([1.0, 2.0])
    v_wd = 0.01 * (grad_wd**2)
    expected_wd = original_data2 - 0.01 * grad_wd / (np.sqrt(v_wd) + 1e-8)

    assert np.allclose(param2.data, expected_wd, rtol=1e-6)

    # Test with multiple parameters
    param3 = Tensor([1.0], requires_grad=True)
    param4 = Tensor([2.0, 3.0], requires_grad=True)
    param3.grad = Tensor([0.5])
    param4.grad = Tensor([0.1, 0.2])

    optimizer_multi = RMSProp([param3, param4], lr=0.01)
    optimizer_multi.step()

    # Both parameters should be updated
    assert not np.array_equal(param3.data, np.array([1.0]))
    assert not np.array_equal(param4.data, np.array([2.0, 3.0]))
    assert len(optimizer_multi.v_buffers) == 2

    # Test zero_grad functionality
    optimizer.zero_grad()
    assert param.grad is None

    print("✅ RMSProp optimizer works correctly!")
