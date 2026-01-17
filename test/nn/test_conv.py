import numpy as np

from tinytorch.nn import Conv2d
from tinytorch.tensor import Tensor


def test_unit_conv2d():
    """🔬 Test Conv2d implementation with multiple configurations."""
    print("🔬 Unit Test: Conv2d...")

    # Test 1: Basic convolution without padding
    print("  Testing basic convolution...")
    conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    x1 = Tensor(np.random.randn(2, 3, 32, 32))
    out1 = conv1(x1)

    expected_h = (32 - 3) + 1  # 30
    expected_w = (32 - 3) + 1  # 30
    assert out1.shape == (2, 16, expected_h, expected_w), (
        f"Expected (2, 16, 30, 30), got {out1.shape}"
    )

    # Test 2: Convolution with padding (same size)
    print("  Testing convolution with padding...")
    conv2 = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x2 = Tensor(np.random.randn(1, 3, 28, 28))
    out2 = conv2(x2)

    # With padding=1, output should be same size as input
    assert out2.shape == (1, 8, 28, 28), f"Expected (1, 8, 28, 28), got {out2.shape}"

    # Test 3: Convolution with stride
    print("  Testing convolution with stride...")
    conv3 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2)
    x3 = Tensor(np.random.randn(1, 1, 16, 16))
    out3 = conv3(x3)

    expected_h = (16 - 3) // 2 + 1  # 7
    expected_w = (16 - 3) // 2 + 1  # 7
    assert out3.shape == (1, 4, expected_h, expected_w), (
        f"Expected (1, 4, 7, 7), got {out3.shape}"
    )

    # Test 4: Parameter counting
    print("  Testing parameter counting...")
    conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=True)
    params = conv4.parameters()

    # Weight: (128, 64, 3, 3) = 73,728 parameters
    # Bias: (128,) = 128 parameters
    # Total: 73,856 parameters
    weight_params = 128 * 64 * 3 * 3
    bias_params = 128
    total_params = weight_params + bias_params

    actual_weight_params = np.prod(conv4.weight.shape)
    actual_bias_params = np.prod(conv4.bias.shape) if conv4.bias is not None else 0
    actual_total = actual_weight_params + actual_bias_params

    assert actual_total == total_params, (
        f"Expected {total_params} parameters, got {actual_total}"
    )
    assert len(params) == 2, f"Expected 2 parameter tensors, got {len(params)}"

    # Test 5: No bias configuration
    print("  Testing no bias configuration...")
    conv5 = Conv2d(in_channels=3, out_channels=16, kernel_size=5, bias=False)
    params5 = conv5.parameters()
    assert len(params5) == 1, (
        f"Expected 1 parameter tensor (no bias), got {len(params5)}"
    )
    assert conv5.bias is None, "Bias should be None when bias=False"

    print("✅ Conv2d works correctly!")
