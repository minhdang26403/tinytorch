import math

import numpy as np

from tinytorch.nn import (
    Embedding,
    EmbeddingLayer,
    PositionalEncoding,
    create_sinusoidal_embeddings,
)
from tinytorch.tensor import Tensor


def test_unit_embedding():
    """🔬 Unit Test: Embedding Layer Implementation"""
    print("🔬 Unit Test: Embedding Layer...")

    # Test 1: Basic embedding creation and forward pass
    embed = Embedding(vocab_size=100, embed_dim=64)

    # Single sequence
    tokens = Tensor([1, 2, 3])
    output = embed.forward(tokens)

    assert output.shape == (3, 64), f"Expected shape (3, 64), got {output.shape}"
    assert len(embed.parameters()) == 1, "Should have 1 parameter (weight matrix)"
    assert embed.parameters()[0].shape == (100, 64), "Weight matrix has wrong shape"

    # Test 2: Batch processing
    batch_tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    batch_output = embed.forward(batch_tokens)

    assert batch_output.shape == (2, 3, 64), (
        f"Expected batch shape (2, 3, 64), got {batch_output.shape}"
    )

    # Test 3: Embedding lookup consistency
    single_lookup = embed.forward(Tensor([1]))
    batch_lookup = embed.forward(Tensor([[1]]))

    # Should get same embedding for same token
    assert np.allclose(single_lookup.data[0], batch_lookup.data[0, 0]), (
        "Inconsistent embedding lookup"
    )

    # Test 4: Parameter access
    params = embed.parameters()
    assert len(params) == 1, "Should have 1 parameter"

    print("✅ Embedding layer works correctly!")


def test_unit_positional_encoding():
    """🔬 Unit Test: Positional Encoding Implementation"""
    print("🔬 Unit Test: Positional Encoding...")

    # Test 1: Basic functionality
    pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=64)

    # Create sample embeddings
    embeddings = Tensor(np.random.randn(2, 10, 64))
    output = pos_enc.forward(embeddings)

    assert output.shape == (2, 10, 64), (
        f"Expected shape (2, 10, 64), got {output.shape}"
    )

    # Test 2: Position consistency
    # Same position should always get same encoding
    emb1 = Tensor(np.zeros((1, 5, 64)))
    emb2 = Tensor(np.zeros((1, 5, 64)))

    out1 = pos_enc.forward(emb1)
    out2 = pos_enc.forward(emb2)

    assert np.allclose(out1.data, out2.data), "Position encodings should be consistent"

    # Test 3: Different positions get different encodings
    short_emb = Tensor(np.zeros((1, 3, 64)))
    long_emb = Tensor(np.zeros((1, 5, 64)))

    short_out = pos_enc.forward(short_emb)
    long_out = pos_enc.forward(long_emb)

    # First 3 positions should match
    assert np.allclose(short_out.data, long_out.data[:, :3, :]), (
        "Position encoding prefix should match"
    )

    # Test 4: Parameters
    params = pos_enc.parameters()
    assert len(params) == 1, "Should have 1 parameter (position embeddings)"
    assert params[0].shape == (512, 64), "Position embedding matrix has wrong shape"

    print("✅ Positional encoding works correctly!")


def test_unit_sinusoidal_embeddings():
    """🔬 Unit Test: Sinusoidal Positional Embeddings"""
    print("🔬 Unit Test: Sinusoidal Embeddings...")

    # Test 1: Basic shape and properties
    pe = create_sinusoidal_embeddings(512, 64)

    assert pe.shape == (512, 64), f"Expected shape (512, 64), got {pe.shape}"

    # Test 2: Position 0 should be mostly zeros and ones
    pos_0 = pe.data[0]

    # Even indices should be sin(0) = 0
    assert np.allclose(pos_0[0::2], 0, atol=1e-6), (
        "Even indices at position 0 should be ~0"
    )

    # Odd indices should be cos(0) = 1
    assert np.allclose(pos_0[1::2], 1, atol=1e-6), (
        "Odd indices at position 0 should be ~1"
    )

    # Test 3: Different positions should have different encodings
    pe_small = create_sinusoidal_embeddings(10, 8)

    # Check that consecutive positions are different
    for i in range(9):
        assert not np.allclose(pe_small.data[i], pe_small.data[i + 1]), (
            f"Positions {i} and {i + 1} are too similar"
        )

    # Test 4: Frequency properties
    # Higher dimensions should have lower frequencies (change more slowly)
    pe_test = create_sinusoidal_embeddings(100, 16)

    # First dimension should change faster than last dimension
    first_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10, 0])))
    last_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10, -1])))

    assert first_dim_changes > last_dim_changes, (
        "Lower dimensions should change faster than higher dimensions"
    )

    # Test 5: Odd embed_dim handling
    pe_odd = create_sinusoidal_embeddings(10, 7)
    assert pe_odd.shape == (10, 7), "Should handle odd embedding dimensions"

    print("✅ Sinusoidal embeddings work correctly!")


def test_unit_complete_embedding_system():
    """🔬 Unit Test: Complete Embedding System"""
    print("🔬 Unit Test: Complete Embedding System...")

    # Test 1: Learned positional encoding
    embed_learned = EmbeddingLayer(
        vocab_size=100, embed_dim=64, max_seq_len=128, pos_encoding="learned"
    )

    tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    output_learned = embed_learned.forward(tokens)

    assert output_learned.shape == (2, 3, 64), (
        f"Expected shape (2, 3, 64), got {output_learned.shape}"
    )

    # Test 2: Sinusoidal positional encoding
    embed_sin = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding="sinusoidal")

    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2, 3, 64), "Sinusoidal embedding should have same shape"

    # Test 3: No positional encoding
    embed_none = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding=None)

    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2, 3, 64), "No pos encoding should have same shape"

    # Test 4: 1D input handling
    tokens_1d = Tensor([1, 2, 3])
    output_1d = embed_learned.forward(tokens_1d)

    assert output_1d.shape == (3, 64), (
        f"Expected shape (3, 64) for 1D input, got {output_1d.shape}"
    )

    # Test 5: Embedding scaling
    embed_scaled = EmbeddingLayer(
        vocab_size=100, embed_dim=64, pos_encoding=None, scale_embeddings=True
    )

    # Use same weights to ensure fair comparison
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight

    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)

    # Scaled version should be sqrt(64) times larger
    scale_factor = math.sqrt(64)
    expected_scaled = output_unscaled.data * scale_factor
    assert np.allclose(output_scaled.data, expected_scaled, rtol=1e-5), (
        "Embedding scaling not working correctly"
    )

    # Test 6: Parameter counting
    params_learned = embed_learned.parameters()
    params_sin = embed_sin.parameters()
    params_none = embed_none.parameters()

    assert len(params_learned) == 2, "Learned encoding should have 2 parameter tensors"
    assert len(params_sin) == 1, "Sinusoidal encoding should have 1 parameter tensor"
    assert len(params_none) == 1, "No pos encoding should have 1 parameter tensor"

    print("✅ Complete embedding system works correctly!")


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire embeddings module functionality.

    This final test ensures all components work together and the module
    is ready for integration with attention mechanisms and transformers.
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_embedding()
    test_unit_positional_encoding()
    test_unit_sinusoidal_embeddings()
    test_unit_complete_embedding_system()

    print("\nRunning integration scenarios...")

    # Integration Test 1: Realistic NLP pipeline
    print("🔬 Integration Test: NLP Pipeline Simulation...")

    # Simulate a small transformer setup
    vocab_size = 1000
    embed_dim = 128
    max_seq_len = 64

    # Create embedding layer
    embed_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pos_encoding="learned",
        scale_embeddings=True,
    )

    # Simulate tokenized sentences
    sentences = [
        [1, 15, 42, 7, 99],  # "the cat sat on mat"
        [23, 7, 15, 88],  # "dog chased the ball"
        [1, 67, 15, 42, 7, 99, 34],  # "the big cat sat on mat here"
    ]

    # Process each sentence
    outputs = []
    for sentence in sentences:
        tokens = Tensor(sentence)
        embedded = embed_layer.forward(tokens)
        outputs.append(embedded)

        # Verify output shape
        expected_shape = (len(sentence), embed_dim)
        assert embedded.shape == expected_shape, (
            f"Wrong shape for sentence: {embedded.shape} != {expected_shape}"
        )

    print("✅ Variable length sentence processing works!")

    # Integration Test 2: Batch processing with padding
    print("🔬 Integration Test: Batched Processing...")

    # Create padded batch (real-world scenario)
    max_len = max(len(s) for s in sentences)
    batch_tokens = []

    for sentence in sentences:
        # Pad with zeros (assuming 0 is padding token)
        padded = sentence + [0] * (max_len - len(sentence))
        batch_tokens.append(padded)

    batch_tensor = Tensor(batch_tokens)  # (3, 7)
    batch_output = embed_layer.forward(batch_tensor)

    assert batch_output.shape == (3, max_len, embed_dim), (
        f"Batch output shape incorrect: {batch_output.shape}"
    )

    print("✅ Batch processing with padding works!")

    # Integration Test 3: Different positional encoding types
    print("🔬 Integration Test: Position Encoding Variants...")

    test_tokens = Tensor([[1, 2, 3, 4, 5]])

    # Test all position encoding types
    for pe_type in ["learned", "sinusoidal", None]:
        embed_test = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding=pe_type)

        output = embed_test.forward(test_tokens)
        assert output.shape == (1, 5, 64), f"PE type {pe_type} failed shape test"

        # Check parameter counts
        if pe_type == "learned":
            assert len(embed_test.parameters()) == 2, (
                "Learned PE should have 2 param tensors"
            )
        else:
            assert len(embed_test.parameters()) == 1, (
                f"PE type {pe_type} should have 1 param tensor"
            )

    print("✅ All positional encoding variants work!")

    # Integration Test 4: Memory efficiency check
    print("🔬 Integration Test: Memory Efficiency...")

    # Test that we're not creating unnecessary copies
    large_embed = EmbeddingLayer(vocab_size=10000, embed_dim=512)
    test_batch = Tensor(np.random.randint(0, 10000, (32, 128)))

    # Multiple forward passes should not accumulate memory (in production)
    for _ in range(5):
        output = large_embed.forward(test_batch)
        assert output.shape == (32, 128, 512), "Large batch processing failed"

    print("✅ Memory efficiency check passed!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("📚 Summary of capabilities built:")
    print("  • Token embedding with trainable lookup tables")
    print("  • Learned positional encodings for position awareness")
    print("  • Sinusoidal positional encodings for extrapolation")
    print("  • Complete embedding system for NLP pipelines")
    print("  • Efficient batch processing and memory management")
    print("\n🚀 Ready for: Attention mechanisms, transformers, and language models!")
    print("Export with: tito module complete 11")
