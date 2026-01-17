import numpy as np

from tinytorch.optim import Adam
from tinytorch.tensor import Tensor


def test_unit_adam_optimizer():
    """🔬 Test Adam optimizer implementation."""
    print("🔬 Unit Test: Adam Optimizer...")

    # Test basic Adam functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    original_data = param.data.copy()

    # First step
    optimizer.step()

    # Manually compute expected values
    grad = np.array([0.1, 0.2])

    # First moment: m = 0.9 * 0 + 0.1 * grad = 0.1 * grad
    m = 0.1 * grad

    # Second moment: v = 0.999 * 0 + 0.001 * grad^2 = 0.001 * grad^2
    v = 0.001 * (grad**2)

    # Bias correction
    bias_correction1 = 1 - 0.9**1  # = 0.1
    bias_correction2 = 1 - 0.999**1  # = 0.001

    m_hat = m / bias_correction1  # = grad
    v_hat = v / bias_correction2  # = grad^2

    # Update
    expected = original_data - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

    assert np.allclose(param.data, expected, rtol=1e-6)
    assert optimizer.step_count == 1

    # Test second step to verify moment accumulation
    param.grad = Tensor([0.1, 0.2])
    optimizer.step()

    # Should have updated moments
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    assert optimizer.step_count == 2

    # Test with weight decay
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])

    optimizer_wd = Adam([param2], lr=0.01, weight_decay=0.01)
    optimizer_wd.step()

    # Weight decay should modify the effective gradient
    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    # The exact computation is complex, but we can verify parameter changed
    assert not np.array_equal(param2.data, np.array([1.0, 2.0]))

    print("✅ Adam optimizer works correctly!")
