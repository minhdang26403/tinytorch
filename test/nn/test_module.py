import numpy as np

from tinytorch.nn import Dropout, Linear, ReLU
from tinytorch.tensor import Tensor

XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)


def test_unit_linear_layer():
    # Test layer creation
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)

    # Test Xavier initialization (weights should be reasonably scaled)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(XAVIER_SCALE_FACTOR / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, (
        f"Weight std {weight_std} not close to Xavier {expected_std}"
    )

    # Test bias initialization (should be zeros)
    assert np.allclose(layer.bias.data, 0), "Bias should be initialized to zeros"

    # Test forward pass
    x = Tensor(np.random.randn(32, 784))  # Batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32, 256), f"Expected shape (32, 256), got {y.shape}"

    # Test no bias option
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1  # Only weight, no bias

    # Test parameters method
    params = layer.parameters()
    assert len(params) == 2  # Weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("✅ Linear layer works correctly!")


def test_edge_cases_linear():
    layer = Linear(10, 5)

    # Test single sample (should handle 2D input)
    x_2d = Tensor(np.random.randn(1, 10))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5), "Should handle single sample"

    # Test zero batch size (edge case)
    x_empty = Tensor(np.random.randn(0, 10))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0, 5), "Should handle empty batch"

    # Test numerical stability with large weights
    layer_large = Linear(10, 5)
    layer_large.weight.data = np.ones((10, 5)) * 100  # Large but not extreme
    x = Tensor(np.ones((1, 10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data)), "Should not produce NaN with large weights"
    assert not np.any(np.isinf(y.data)), "Should not produce Inf with large weights"

    # Test with no bias
    layer_no_bias = Linear(10, 5, bias=False)
    x = Tensor(np.random.randn(4, 10))
    y = layer_no_bias.forward(x)
    assert y.shape == (4, 5), "Should work without bias"


def test_parameter_collection_linear():
    layer = Linear(10, 5)

    # Verify parameter collection works
    params = layer.parameters()
    assert len(params) == 2, "Should return 2 parameters (weight and bias)"
    assert params[0].shape == (10, 5), "First param should be weight"
    assert params[1].shape == (5,), "Second param should be bias"

    # Test layer without bias
    layer_no_bias = Linear(10, 5, bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, "Should return 1 parameter (weight only)"


def test_unit_dropout_layer():
    # Test dropout creation
    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    # Test inference mode (should pass through unchanged)
    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data), (
        "Inference should pass through unchanged"
    )

    # Test training mode with zero dropout (should pass through unchanged)
    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert np.array_equal(x.data, y_zero.data), (
        "Zero dropout should pass through unchanged"
    )

    # Test training mode with full dropout (should zero everything)
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert np.allclose(y_full.data, 0), "Full dropout should zero everything"

    # Test training mode with partial dropout
    # Note: This is probabilistic, so we test statistical properties
    np.random.seed(42)  # For reproducible test
    x_large = Tensor(np.ones((1000,)))  # Large tensor for statistical significance
    y_train = dropout.forward(x_large, training=True)

    # Count non-zero elements (approximately 50% should survive)
    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500
    # Use 3-sigma bounds: std = sqrt(n*p*(1-p)) = sqrt(1000*0.5*0.5) ≈ 15.8
    std_error = np.sqrt(1000 * 0.5 * 0.5)
    lower_bound = expected - 3 * std_error  # ≈ 453
    upper_bound = expected + 3 * std_error  # ≈ 547
    assert lower_bound < non_zero_count < upper_bound, (
        f"Expected {expected}±{3 * std_error:.0f} survivors, got {non_zero_count}"
    )

    # Test scaling (surviving elements should be scaled by 1/(1-p) = 2.0)
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0  # 1.0 / (1 - 0.5)
    assert np.allclose(surviving_values, expected_value), (
        f"Surviving values should be {expected_value}"
    )

    # Test no parameters
    params = dropout.parameters()
    assert len(params) == 0, "Dropout should have no parameters"

    # Test invalid probability
    try:
        Dropout(-0.1)
        assert False, "Should raise ValueError for negative probability"
    except ValueError:
        pass

    try:
        Dropout(1.1)
        assert False, "Should raise ValueError for probability > 1"
    except ValueError:
        pass


def analyze_layer_memory():
    """Analyze memory usage patterns in layer operations."""

    # Test different layer sizes
    layer_configs = [
        (784, 256),  # MNIST → hidden
        (256, 256),  # Hidden → hidden
        (256, 10),  # Hidden → output
        (2048, 2048),  # Large hidden
    ]

    print("\nLinear Layer Memory Analysis:")
    print("Configuration → Weight Memory → Bias Memory → Total Memory")

    for in_feat, out_feat in layer_configs:
        # Calculate memory usage
        weight_memory = in_feat * out_feat * 4  # 4 bytes per float32
        bias_memory = out_feat * 4
        total_memory = weight_memory + bias_memory

        print(
            f"({in_feat:4d}, {out_feat:4d}) → {weight_memory / 1024:7.1f} KB → {bias_memory / 1024:6.1f} KB → {total_memory / 1024:7.1f} KB"
        )

    # Analyze multi-layer memory scaling
    print("\nMulti-layer Model Memory Scaling:")
    hidden_sizes = [128, 256, 512, 1024, 2048]

    for hidden_size in hidden_sizes:
        # 3-layer MLP: 784 → hidden → hidden/2 → 10
        layer1_params = 784 * hidden_size + hidden_size
        layer2_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        layer3_params = (hidden_size // 2) * 10 + 10

        total_params = layer1_params + layer2_params + layer3_params
        memory_mb = total_params * 4 / (1024 * 1024)

        print(
            f"Hidden={hidden_size:4d}: {total_params:7,} params = {memory_mb:5.1f} MB"
        )


def analyze_layer_performance():
    """Analyze computational complexity of layer operations."""
    import time

    # Test forward pass FLOPs
    batch_sizes = [1, 32, 128, 512]
    layer = Linear(784, 256)

    print("\nLinear Layer FLOPs Analysis:")
    print("Batch Size → Matrix Multiply FLOPs → Bias Add FLOPs → Total FLOPs")

    for batch_size in batch_sizes:
        # Matrix multiplication: (batch, in) @ (in, out) = batch * in * out FLOPs
        matmul_flops = batch_size * 784 * 256
        # Bias addition: batch * out FLOPs
        bias_flops = batch_size * 256
        total_flops = matmul_flops + bias_flops

        print(
            f"{batch_size:10d} → {matmul_flops:15,} → {bias_flops:13,} → {total_flops:11,}"
        )

    # Add timing measurements
    print("\nLinear Layer Timing Analysis:")
    print("Batch Size → Time (ms) → Throughput (samples/sec)")

    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size, 784))

        # Warm up
        for _ in range(10):
            _ = layer.forward(x)

        # Time multiple iterations
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer.forward(x)
        elapsed = time.perf_counter() - start

        time_per_forward = (elapsed / iterations) * 1000  # Convert to ms
        throughput = (batch_size * iterations) / elapsed

        print(
            f"{batch_size:10d} → {time_per_forward:8.3f} ms → {throughput:12,.0f} samples/sec"
        )

    print("\n💡 Key Insights:")
    print("🚀 Linear layer complexity: O(batch_size × in_features × out_features)")
    print("🚀 Memory grows linearly with batch size, quadratically with layer width")
    print("🚀 Dropout adds minimal computational overhead (element-wise operations)")
    print("🚀 Larger batches amortize overhead, improving throughput efficiency")


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_linear_layer()
    test_edge_cases_linear()
    test_parameter_collection_linear()
    test_unit_dropout_layer()

    print("\nRunning integration scenarios...")

    # Test realistic neural network construction with manual composition
    print("🔬 Integration Test: Multi-layer Network...")

    # Use ReLU imported from package at module level
    ReLU_class = ReLU

    # Build individual layers for manual composition
    layer1 = Linear(784, 128)
    activation1 = ReLU_class()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU_class()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)

    # Test end-to-end forward pass with manual composition
    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 784))

    # Manual forward pass
    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    output = layer3.forward(x)

    assert output.shape == (batch_size, 10), (
        f"Expected output shape ({batch_size}, 10), got {output.shape}"
    )

    # Test parameter counting from individual layers
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6  # 3 weights + 3 biases from 3 Linear layers
    assert len(all_params) == expected_params, (
        f"Expected {expected_params} parameters, got {len(all_params)}"
    )

    # Test individual layer functionality
    test_x = Tensor(np.random.randn(4, 784))
    # Test dropout in training vs inference
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x, training=True)
    infer_output = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, infer_output.data), (
        "Inference mode should pass through unchanged"
    )

    print("✅ Multi-layer network integration works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 03_layers")


if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 03: LAYERS - COMPREHENSIVE VALIDATION")
    print("=" * 70)

    # Run module integration test
    test_module()

    print("\n" + "=" * 70)
    print("SYSTEMS ANALYSIS")
    print("=" * 70)

    # Run analysis functions
    analyze_layer_memory()
    print("\n")
    analyze_layer_performance()

    print("\n" + "=" * 70)
    print("✅ MODULE 03 COMPLETE!")
    print("=" * 70)
