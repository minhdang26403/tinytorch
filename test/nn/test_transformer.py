import numpy as np

from tinytorch.nn import GPT, MLP, LayerNorm, TransformerBlock
from tinytorch.tensor import Tensor


def test_unit_layer_norm():
    """🔬 Test LayerNorm implementation."""
    print("🔬 Unit Test: Layer Normalization...")

    # Test basic normalization
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # (2, 4)

    normalized = ln.forward(x)

    # Check output shape
    assert normalized.shape == (2, 4)

    # Check normalization properties (approximately)
    # For each sample, mean should be close to 0, std close to 1
    for i in range(2):
        sample_mean = np.mean(normalized.data[i])
        sample_std = np.std(normalized.data[i])
        assert abs(sample_mean) < 1e-5, f"Mean should be ~0, got {sample_mean}"
        assert abs(sample_std - 1.0) < 1e-4, f"Std should be ~1, got {sample_std}"

    # Test parameter shapes
    params = ln.parameters()
    assert len(params) == 2
    assert params[0].shape == (4,)  # gamma
    assert params[1].shape == (4,)  # beta

    print("✅ LayerNorm works correctly!")


def test_unit_mlp():
    """🔬 Test MLP implementation."""
    print("🔬 Unit Test: MLP (Feed-Forward Network)...")

    # Test MLP with standard 4x expansion
    embed_dim = 64
    mlp = MLP(embed_dim)

    # Test forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = mlp.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Check hidden dimension is 4x
    assert mlp.hidden_dim == 4 * embed_dim

    # Test parameter counting
    params = mlp.parameters()
    expected_params = 4  # 2 weights + 2 biases
    assert len(params) == expected_params

    # Test custom hidden dimension
    custom_mlp = MLP(embed_dim, hidden_dim=128)
    assert custom_mlp.hidden_dim == 128

    print("✅ MLP works correctly!")


def test_unit_transformer_block():
    """🔬 Test TransformerBlock implementation."""
    print("🔬 Unit Test: Transformer Block...")

    # Test transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = block.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Test with causal mask (for autoregressive generation)
    mask = Tensor(np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1))
    masked_output = block.forward(x, mask)
    assert masked_output.shape == (batch_size, seq_len, embed_dim)

    # Test parameter counting
    params = block.parameters()
    expected_components = 4  # attention, ln1, ln2, mlp parameters
    assert (
        len(params) > expected_components
    )  # Should have parameters from all components

    # Test different configurations
    large_block = TransformerBlock(embed_dim=128, num_heads=8, mlp_ratio=2)
    assert large_block.mlp.hidden_dim == 256  # 128 * 2

    print("✅ TransformerBlock works correctly!")


def test_unit_gpt():
    """🔬 Test GPT model implementation."""
    print("🔬 Unit Test: GPT Model...")

    # Test small GPT model
    vocab_size = 100
    embed_dim = 64
    num_layers = 2
    num_heads = 4

    model = GPT(vocab_size, embed_dim, num_layers, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    logits = model.forward(tokens)

    # Check output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape

    # Test generation
    prompt = Tensor(np.random.randint(0, vocab_size, (1, 5)))
    generated = model.generate(prompt, max_new_tokens=3)

    # Check generation shape
    assert generated.shape == (1, 8)  # 5 prompt + 3 new tokens

    # Test parameter counting
    params = model.parameters()
    assert len(params) > 10  # Should have many parameters from all components

    # Test different model sizes
    larger_model = GPT(vocab_size=200, embed_dim=128, num_layers=4, num_heads=8)
    test_tokens = Tensor(np.random.randint(0, 200, (1, 10)))
    larger_logits = larger_model.forward(test_tokens)
    assert larger_logits.shape == (1, 10, 200)

    print("✅ GPT model works correctly!")
