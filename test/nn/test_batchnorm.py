import numpy as np

from tinytorch.nn import BatchNorm2d
from tinytorch.tensor import Tensor


def test_unit_batchnorm2d():
    """🔬 Test BatchNorm2d implementation."""
    print("🔬 Unit Test: BatchNorm2d...")

    # Test 1: Basic forward pass shape
    print("  Testing basic forward pass...")
    bn = BatchNorm2d(num_features=16)
    x = Tensor(np.random.randn(4, 16, 8, 8))  # batch=4, channels=16, 8x8
    y = bn(x)

    assert y.shape == x.shape, f"Output shape should match input, got {y.shape}"

    # Test 2: Training mode normalization
    print("  Testing training mode normalization...")
    bn2 = BatchNorm2d(num_features=8)
    bn2.train()  # Ensure training mode

    # Create input with known statistics per channel
    x2 = Tensor(np.random.randn(32, 8, 4, 4) * 10 + 5)  # Mean~5, std~10
    y2 = bn2(x2)

    # After normalization, each channel should have mean≈0, std≈1
    # (before gamma/beta are applied, since gamma=1, beta=0)
    for c in range(8):
        channel_mean = np.mean(y2.data[:, c, :, :])
        channel_std = np.std(y2.data[:, c, :, :])
        assert abs(channel_mean) < 0.1, (
            f"Channel {c} mean should be ~0, got {channel_mean:.3f}"
        )
        assert abs(channel_std - 1.0) < 0.1, (
            f"Channel {c} std should be ~1, got {channel_std:.3f}"
        )

    # Test 3: Running statistics update
    print("  Testing running statistics update...")
    initial_running_mean = bn2.running_mean.copy()

    # Forward pass updates running stats
    x3 = Tensor(np.random.randn(16, 8, 4, 4) + 3)  # Offset mean
    _ = bn2(x3)

    # Running mean should have moved toward batch mean
    assert not np.allclose(bn2.running_mean, initial_running_mean), (
        "Running mean should update during training"
    )

    # Test 4: Eval mode uses running statistics
    print("  Testing eval mode behavior...")
    bn3 = BatchNorm2d(num_features=4)

    # Train on some data to establish running stats
    for _ in range(10):
        x_train = Tensor(np.random.randn(8, 4, 4, 4) * 2 + 1)
        _ = bn3(x_train)

    saved_running_mean = bn3.running_mean.copy()
    saved_running_var = bn3.running_var.copy()

    # Switch to eval mode
    bn3.eval()

    # Process different data - running stats should NOT change
    x_eval = Tensor(np.random.randn(2, 4, 4, 4) * 5)  # Different distribution
    _ = bn3(x_eval)

    assert np.allclose(bn3.running_mean, saved_running_mean), (
        "Running mean should not change in eval mode"
    )
    assert np.allclose(bn3.running_var, saved_running_var), (
        "Running var should not change in eval mode"
    )

    # Test 5: Parameter counting
    print("  Testing parameter counting...")
    bn4 = BatchNorm2d(num_features=64)
    params = bn4.parameters()

    assert len(params) == 2, (
        f"Should have 2 parameters (gamma, beta), got {len(params)}"
    )
    assert params[0].shape == (64,), (
        f"Gamma shape should be (64,), got {params[0].shape}"
    )
    assert params[1].shape == (64,), (
        f"Beta shape should be (64,), got {params[1].shape}"
    )

    print("✅ BatchNorm2d works correctly!")
