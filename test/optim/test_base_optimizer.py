from tinytorch.optim import Optimizer
from tinytorch.tensor import Tensor


def test_unit_optimizer_base():
    """🔬 Test base Optimizer functionality."""
    print("🔬 Unit Test: Base Optimizer...")

    # Create test parameters
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    # Add some gradients
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([[0.3, 0.4], [0.5, 0.6]])

    # Create optimizer
    optimizer = Optimizer([param1, param2])

    # Test parameter storage
    assert len(optimizer.params) == 2
    assert optimizer.params[0] is param1
    assert optimizer.params[1] is param2
    assert optimizer.step_count == 0

    # Test zero_grad
    optimizer.zero_grad()
    assert param1.grad is None
    assert param2.grad is None

    # Test that optimizer accepts any tensor (no validation required)
    # Gradient tracking is handled by the autograd module
    regular_param = Tensor([1.0])
    opt = Optimizer([regular_param])
    assert len(opt.params) == 1

    print("✅ Base Optimizer works correctly!")
