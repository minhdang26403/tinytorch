import numpy as np

from tinytorch.nn.utils import clip_grad_norm
from tinytorch.tensor import Tensor


def test_unit_clip_grad_norm():
    """🔬 Test clip_grad_norm implementation."""
    print("🔬 Unit Test: Gradient Clipping...")

    # Use real Tensor from Module 01
    # Tensor already imported at module level

    # Test case 1: Large gradients that need clipping
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param1.grad = np.array([3.0, 4.0])  # norm = 5.0

    param2 = Tensor([3.0, 4.0], requires_grad=True)
    param2.grad = np.array([6.0, 8.0])  # norm = 10.0

    params = [param1, param2]
    # Total norm = sqrt(5² + 10²) = sqrt(125) ≈ 11.18

    original_norm = clip_grad_norm(params, max_norm=1.0)

    # Check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"

    # Check gradients were clipped
    new_norm = 0.0
    for param in params:
        if isinstance(param.grad, Tensor):
            # Trust that Tensor has .data attribute
            grad_data = param.grad.data
        else:
            grad_data = param.grad
        new_norm += np.sum(grad_data**2)
    new_norm = np.sqrt(new_norm)

    print(f"Original norm: {original_norm:.2f}")
    print(f"Clipped norm: {new_norm:.2f}")

    assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1.0, got {new_norm}"

    # Test case 2: Small gradients that don't need clipping
    small_param = Tensor([1.0, 2.0], requires_grad=True)
    small_param.grad = np.array([0.1, 0.2])
    small_params = [small_param]
    original_small = clip_grad_norm(small_params, max_norm=1.0)

    assert original_small < 1.0, "Small gradients shouldn't be clipped"

    print("✅ Gradient clipping works correctly!")
