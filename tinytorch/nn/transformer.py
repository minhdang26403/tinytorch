import numpy as np

from tinytorch.attention import MultiHeadAttention
from tinytorch.tensor import Tensor

from .activation import GELU
from .embedding import EmbeddingLayer
from .linear import Linear


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal (autoregressive) attention mask.

    This mask ensures that position i can only attend to positions j where j ≤ i.
    Essential for autoregressive language models like GPT.

    Args:
        seq_len: Length of the sequence

    Returns:
        Tensor of shape (1, seq_len, seq_len) with:
        - 1.0 for positions that CAN be attended to (lower triangle)
        - 0.0 for positions that CANNOT be attended to (upper triangle)

    Example:
        For seq_len=4, creates:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Usage:
        >>> from tinytorch.core.transformers import create_causal_mask
        >>> mask = create_causal_mask(seq_len=10)
        >>> output = attention(x, mask=mask)
    """
    # Lower triangular matrix: 1 = can attend, 0 = cannot attend
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension


class LayerNorm:
    """
    Layer Normalization for transformer blocks.

    Normalizes across the feature dimension (last axis) for each sample independently,
    unlike batch normalization which normalizes across the batch dimension.
    """

    def __init__(self, normalized_shape, eps=1e-56):
        """
        Initialize LayerNorm with learnable parameters.

        EXAMPLE:
        >>> ln = LayerNorm(512)  # For 512-dimensional embeddings
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, features)
        >>> normalized = ln.forward(x)
        >>> # Each (2, 10) sample normalized independently across 512 features
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: scale and shift
        self.gamma = np.ones(self.normalized_shape)  # Scale parameter
        self.beta = np.zeros(self.normalized_shape)  # Shift parameter

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.

        APPROACH:
        1. Compute mean and variance across the last dimension
        2. Normalize: (x - mean) / sqrt(variance + eps)
        3. Apply learnable scale and shift: gamma * normalized + beta
        """
        # Compute statistics across last dimension (features)
        x_data = x.data
        mean = np.mean(x_data, axis=-1, keepdims=True)
        diff = x_data - mean

        # Compute variance: E[(x - μ)²]
        variance = (diff * diff).mean(axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)

        # Normalize
        normalized = diff / std

        # Apply learnable transformation
        output = normalized * self.gamma + self.beta
        return Tensor(output)

    def __call__(self, x):
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear -> GELU -> Linear with expansion ratio of 4:1.
    This provides the non-linear transformation in each transformer block.
    """

    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1):
        """
        Initialize MLP with two linear layers.

        EXAMPLE:
        >>> mlp = MLP(512)  # Will create 512 -> 2048 -> 512 network
        >>> x = Tensor(np.random.randn(2, 10, 512))
        >>> output = mlp.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer feed-forward network
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()  # Use GELU activation from activations module
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through MLP.
        """
        # First linear layer with expansion
        hidden = self.linear1.forward(x)

        # GELU activation (YOUR activation from Module 03!)
        hidden = self.gelu.forward(hidden)

        # Second linear layer back to original size
        output = self.linear2.forward(hidden)

        return output

    def __call__(self, x):
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params


class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP, and residual connections.

    This is the core building block of GPT and other transformer models.
    Each block processes the input sequence and passes it to the next block.
    """

    def __init__(
        self, embed_dim, num_heads, mlp_ratio=4, ff_dim=None, dropout_prob=0.1
    ):
        """
        Initialize a complete transformer block.

        APPROACH:
        1. Multi-head self-attention for sequence modeling
        2. First layer normalization (pre-norm architecture)
        3. MLP with specified expansion ratio (or explicit ff_dim)
        4. Second layer normalization

        TRANSFORMER BLOCK ARCHITECTURE:
        x → LayerNorm → MultiHeadAttention → + (residual) →
            LayerNorm → MLP → + (residual) → output

        EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalizations (pre-norm architecture)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        # Feed-forward network
        # Support both mlp_ratio and explicit ff_dim for backward compatibility
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        APPROACH:
        1. Apply layer norm, then self-attention, then add residual
        2. Apply layer norm, then MLP, then add residual
        3. Return the transformed sequence

        COMPUTATION FLOW:
        x → ln1 → attention → + x → ln2 → mlp → + → output

        RESIDUAL CONNECTIONS:
        These are crucial for training deep networks - they allow gradients
        to flow directly through the network during backpropagation.
        """
        # First sub-layer: Multi-head self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        normed1 = self.ln1.forward(x)
        # Self-attention: query, key, value are all the same (normed1)
        attention_out = self.attention.forward(normed1, mask)

        # Residual connection
        x = x + attention_out

        # Second sub-layer: MLP with residual connection
        # Pre-norm: LayerNorm before MLP
        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)

        # Residual connection
        output = x + mlp_out

        return output

    def __call__(self, x, mask=None):
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params


class GPT:
    """
    Complete GPT (Generative Pre-trained Transformer) model.

    This combines embeddings, positional encoding, multiple transformer blocks,
    and a language modeling head for text generation.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        """
        Initialize complete GPT model.

        APPROACH:
        1. Token embedding layer to convert tokens to vectors
        2. Positional embedding to add position information
        3. Stack of transformer blocks (the main computation)
        4. Final layer norm and language modeling head

        GPT ARCHITECTURE:
        tokens → embedding → + pos_embedding →
                transformer_blocks → layer_norm → lm_head → logits

        EXAMPLE:
        >>> model = GPT(vocab_size=1000, embed_dim=256, num_layers=6, num_heads=8)
        >>> tokens = Tensor(np.random.randint(0, 1000, (2, 10)))  # (batch, seq)
        >>> logits = model.forward(tokens)
        >>> assert logits.shape == (2, 10, 1000)  # (batch, seq, vocab)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.embedding = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

        # Stack of transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads)
            self.blocks.append(block)

        # Final layer normalization
        self.ln_f = LayerNorm(embed_dim)

        # Language modeling head (projects to vocabulary)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens):
        """
        Forward pass through GPT model.

        APPROACH:
        1. Get token embeddings and positional embeddings
        2. Add them together (broadcasting handles different shapes)
        3. Pass through all transformer blocks sequentially
        4. Apply final layer norm and language modeling head

        COMPUTATION FLOW:
        tokens → embed + pos_embed → blocks → ln_f → lm_head → logits

        CAUSAL MASKING:
        For autoregressive generation, we need to prevent tokens from
        seeing future tokens. This is handled by the attention mask.
        """
        batch_size, seq_len = tokens.shape

        # Get token embeddings and positional embeddings
        x = self.embedding.forward(tokens)

        # Create causal mask for autoregressive generation
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Language modeling head
        logits = self.lm_head.forward(x)

        return logits

    def __call__(self, tokens):
        """Allows the GPT model to be called like a function."""
        return self.forward(tokens)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""
        # Upper triangular matrix filled with -inf
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        return Tensor(mask)

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively.

        APPROACH:
        1. Start with prompt tokens
        2. For each new position:
           - Run forward pass to get logits
           - Sample next token from logits
           - Append to sequence
        3. Return generated sequence

        AUTOREGRESSIVE GENERATION:
        At each step, the model predicts the next token based on all
        previous tokens. This is how GPT generates coherent text.

        EXAMPLE:
        >>> model = GPT(vocab_size=100, embed_dim=64, num_layers=2, num_heads=4)
        >>> prompt = Tensor([[1, 2, 3]])  # Some token sequence
        >>> generated = model.generate(prompt, max_new_tokens=5)
        >>> assert generated.shape[1] == 3 + 5  # original + new tokens
        """
        current_tokens = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(current_tokens)

            # Get logits for last position (next token prediction)
            last_logits = logits.data[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature scaling
            scaled_logits = last_logits / temperature

            # Convert to probabilities (softmax)
            exp_logits = np.exp(
                scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
            )
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Sample next token
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])

            # Append to sequence
            current_tokens = Tensor(
                np.concatenate([current_tokens.data, next_token], axis=1)
            )

        return current_tokens

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())

        return params
