import time

import numpy as np

from tinytorch.attention import MultiHeadAttention, scaled_dot_product_attention
from tinytorch.tensor import Tensor


def test_unit_scaled_dot_product_attention():
    """🔬 Unit Test: Scaled Dot-Product Attention"""
    print("🔬 Unit Test: Scaled Dot-Product Attention...")

    # Test basic functionality
    batch_size, seq_len, d_model = 2, 4, 8
    Q = Tensor(np.random.randn(batch_size, seq_len, d_model))
    K = Tensor(np.random.randn(batch_size, seq_len, d_model))
    V = Tensor(np.random.randn(batch_size, seq_len, d_model))

    output, weights = scaled_dot_product_attention(Q, K, V)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Output shape {output.shape} incorrect"
    )
    assert weights.shape == (batch_size, seq_len, seq_len), (
        f"Weights shape {weights.shape} incorrect"
    )

    # Check attention weights sum to 1 (probability distribution)
    weights_sum = weights.data.sum(axis=2)  # Sum over last dimension
    expected_sum = np.ones((batch_size, seq_len))
    assert np.allclose(weights_sum, expected_sum, atol=1e-6), (
        "Attention weights don't sum to 1"
    )

    # Test with causal mask
    mask = Tensor(
        np.tril(np.ones((batch_size, seq_len, seq_len)), k=0)
    )  # Lower triangular
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)

    # Check that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):  # Future positions
                assert abs(weights_masked.data[b, i, j]) < 1e-6, (
                    f"Future attention not masked at ({i},{j})"
                )

    print("✅ scaled_dot_product_attention works correctly!")


def test_unit_multihead_attention():
    """🔬 Unit Test: Multi-Head Attention"""
    print("🔬 Unit Test: Multi-Head Attention...")

    # Test initialization
    embed_dim, num_heads = 64, 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Check configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    # Test parameter counting (4 linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers × 2), got {len(params)}"

    # Test forward pass
    batch_size, seq_len = 2, 6
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha.forward(x)

    # Check output shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim), (
        f"Output shape {output.shape} incorrect"
    )

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len))))
    output_masked = mha.forward(x, mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

    # Test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4)
    x_small = Tensor(np.random.randn(1, 5, 32))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1, 5, 32)

    print("✅ MultiHeadAttention works correctly!")


def analyze_attention_complexity():
    """📊 Analyze attention computational complexity and memory scaling."""
    print("📊 Analyzing Attention Complexity...")

    # Test different sequence lengths to show O(n²) scaling
    embed_dim = 64
    sequence_lengths = [16, 32, 64, 128, 256]

    print("\nSequence Length vs Attention Matrix Size:")
    print("Seq Len | Attention Matrix | Memory (KB) | Complexity")
    print("-" * 55)

    for seq_len in sequence_lengths:
        # Calculate attention matrix size
        attention_matrix_size = seq_len * seq_len

        # Memory for attention weights (float32 = 4 bytes)
        attention_memory_kb = (attention_matrix_size * 4) / 1024

        # Total complexity (Q@K + softmax + weights@V)
        complexity = 2 * seq_len * seq_len * embed_dim + seq_len * seq_len

        print(
            f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {complexity:10.0f}"
        )

    print("\n💡 Attention memory scales as O(n²) with sequence length")
    print(
        f"🚀 For seq_len=1024, attention matrix alone needs {(1024 * 1024 * 4) / 1024 / 1024:.1f} MB"
    )


def analyze_attention_timing():
    """📊 Measure attention computation time vs sequence length."""
    print("\n📊 Analyzing Attention Timing...")

    embed_dim, num_heads = 64, 8
    sequence_lengths = [32, 64, 128, 256]

    print("\nSequence Length vs Computation Time:")
    print("Seq Len | Time (ms) | Ops/sec | Scaling")
    print("-" * 40)

    prev_time = None
    for seq_len in sequence_lengths:
        # Create test input
        x = Tensor(np.random.randn(1, seq_len, embed_dim))
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = mha.forward(x)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        ops_per_sec = 1000 / avg_time if avg_time > 0 else 0

        # Calculate scaling factor vs previous
        scaling = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:7d} | {avg_time:8.2f} | {ops_per_sec:7.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    print("\n💡 Attention time scales roughly as O(n²) with sequence length")
    print(
        "🚀 This is why efficient attention (FlashAttention) is crucial for long sequences"
    )


def analyze_attention_memory_overhead():
    """📊 Analyze memory overhead during training (forward + backward passes)."""
    print("\n📊 Analyzing Attention Memory Overhead During Training...")

    embed_dim, num_heads = 128, 8
    sequence_lengths = [128, 256, 512, 1024]

    print("\nMemory Overhead Analysis (Training vs Inference):")
    print("Seq Len | Forward | + Gradients | + Optimizer | Total Memory")
    print("-" * 65)

    for seq_len in sequence_lengths:
        # Forward pass memory (attention matrix)
        attention_matrix_mb = (seq_len * seq_len * 4) / (1024 * 1024)

        # Backward pass adds gradient storage (2× forward)
        backward_memory_mb = 2 * attention_matrix_mb

        # Optimizer state (Adam: +2× for momentum and velocity)
        optimizer_memory_mb = backward_memory_mb + 2 * attention_matrix_mb

        print(
            f"{seq_len:7d} | {attention_matrix_mb:6.2f}MB | {backward_memory_mb:10.2f}"
            f"MB | {optimizer_memory_mb:10.2f}MB | {optimizer_memory_mb:11.2f}MB"
        )

    print(
        "\n💡 Training requires 4× memory of inference (forward + grad + 2× optimizer state)"
    )
    print("🚀 For GPT-3 (96 layers, 2048 context): ~6GB just for attention gradients!")


def test_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🔬 Testing Attention Scenarios...")

    # Scenario 1: Small transformer block setup
    print("\n1. Small Transformer Setup:")
    embed_dim, num_heads, seq_len = 128, 8, 32

    # Create embeddings (simulating token embeddings + positional)
    embeddings = Tensor(np.random.randn(2, seq_len, embed_dim))

    # Multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    attended = mha.forward(embeddings)

    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {attended.shape}")
    print(f"   Parameters: {len(mha.parameters())} tensors")

    # Scenario 2: Causal language modeling
    print("\n2. Causal Language Modeling:")

    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    mask = Tensor(np.broadcast_to(causal_mask, (2, seq_len, seq_len)))

    # Apply causal attention
    causal_output = mha.forward(embeddings, mask)

    print(f"   Masked output shape: {causal_output.shape}")
    print(f"   Causal mask applied: {mask.shape}")

    # Scenario 3: Compare attention patterns
    print("\n3. Attention Pattern Analysis:")

    # Create simple test sequence
    simple_embed = Tensor(np.random.randn(1, 4, 16))
    simple_mha = MultiHeadAttention(16, 4)

    # Get attention weights by calling the base function
    Q = simple_mha.q_proj.forward(simple_embed)
    K = simple_mha.k_proj.forward(simple_embed)
    V = simple_mha.v_proj.forward(simple_embed)

    # Reshape for single head analysis
    Q_head = Tensor(Q.data[:, :, :4])  # First head only
    K_head = Tensor(K.data[:, :, :4])
    V_head = Tensor(V.data[:, :, :4])

    _, weights = scaled_dot_product_attention(Q_head, K_head, V_head)

    print(f"   Attention weights shape: {weights.shape}")
    print("   Attention weights (first batch, 4x4 matrix):")
    weight_matrix = weights.data[0, :, :].round(3)

    # Format the attention matrix nicely
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(
            f"{weight_matrix[i, j]:5.3f}" for j in range(4)
        )
        print(row_str)

    print(f"   Row sums: {weights.data[0].sum(axis=1).round(3)} (should be ~1.0)")

    # Scenario 4: Attention with masking visualization
    print("\n4. Causal Masking Effect:")

    # Apply causal mask to the simple example
    simple_mask = Tensor(np.tril(np.ones((1, 4, 4))))
    _, masked_weights = scaled_dot_product_attention(
        Q_head, K_head, V_head, simple_mask
    )

    print("   Causal attention matrix (lower triangular):")
    masked_matrix = masked_weights.data[0, :, :].round(3)
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(
            f"{masked_matrix[i, j]:5.3f}" for j in range(4)
        )
        print(row_str)

    print("   Notice: Upper triangle is zero (can't attend to future)")

    print("\n✅ All attention scenarios work correctly!")


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire attention module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING ATTENTION MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_scaled_dot_product_attention()
    test_unit_multihead_attention()

    print("\nRunning integration scenarios...")
    test_attention_scenarios()

    print("\nRunning performance analysis...")
    analyze_attention_complexity()
    print("\nRunning memory overhead analysis...")
    analyze_attention_memory_overhead()

    print("\n" + "=" * 50)
    print("✅ ATTENTION MODULE COMPLETE! ALL TESTS PASSED!")
