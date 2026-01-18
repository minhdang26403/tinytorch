import numpy as np

from tinytorch.nn import Linear, Softmax
from tinytorch.tensor import Tensor

# Constants for attention computation
MASK_VALUE = (
    -1e9
)  # Large negative value used for attention masking (becomes ~0 after softmax)


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
        mask: Optional causal mask, True=allow, False=mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attended values (batch_size, seq_len, d_model)
        attention_weights: Attention matrix (batch_size, seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(2, 4, 64))  # batch=2, seq=4, dim=64
    >>> K = Tensor(np.random.randn(2, 4, 64))
    >>> V = Tensor(np.random.randn(2, 4, 64))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)  # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)
    >>> print(weights.data[0].sum(axis=1))  # Each row sums to ~1.0
    """

    d_model = Q.shape[-1]

    # (batch_size, seq_len, seq_len)
    attention_scores = Q @ K.transpose(-2, -1)
    attention_scores = attention_scores / np.sqrt(d_model)

    if mask is not None:
        # Value 0 in mask indicates mask while value 1 indicates no mask, so we do
        # (1 - mask) and multiply with -1e9 to prepare for softmax layer
        mask_data = (1 - mask.data) * MASK_VALUE
        attention_scores = attention_scores + Tensor(mask_data)

    attention_weights = Softmax(dim=-1).forward(attention_scores)
    # (batch_size, seq_len, d_model)
    output = attention_weights @ V

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel, each learning different relationships.
    This is the core component of transformer architectures.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize multi-head attention.

        TODO: Set up linear projections and validate configuration

        APPROACH:
        1. Validate that embed_dim is divisible by num_heads
        2. Calculate head_dim (embed_dim // num_heads)
        3. Create linear layers for Q, K, V projections
        4. Create output projection layer
        5. Store configuration parameters

        Args:
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of parallel attention heads

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> mha.head_dim  # 64 (512 / 8)
        >>> len(mha.parameters())  # 4 linear layers * 2 params each = 8 tensors

        HINTS:
        - head_dim = embed_dim // num_heads must be integer
        - Need 4 Linear layers: q_proj, k_proj, v_proj, out_proj
        - Each projection maps embed_dim → embed_dim
        """
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).\n"
                f"  Issue: Multi-head attention splits embed_dim into num_heads heads.\n"
                f"  Fix: Choose embed_dim and num_heads such that embed_dim % num_heads == 0.\n"
                f"  Example: embed_dim=512, num_heads=8 works (512/8=64 per head)."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass through multi-head attention.

        APPROACH:
        1. Extract input dimensions (batch_size, seq_len, embed_dim)
        2. Project input to Q, K, V using linear layers
        3. Reshape projections to separate heads: (batch, seq, heads, head_dim)
        4. Transpose to (batch, heads, seq, head_dim) for parallel processing
        5. Apply scaled dot-product attention to each head
        6. Transpose back and reshape to merge heads
        7. Apply output projection

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            output: Attended representation (batch_size, seq_len, embed_dim)

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10, dim=64
        >>> output = mha.forward(x)
        >>> print(output.shape)  # (2, 10, 64) - same as input

        HINTS:
        - Reshape: (batch, seq, embed_dim) → (batch, seq, heads, head_dim)
        - Transpose: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        - After attention: reverse the process to merge heads
        - Use scaled_dot_product_attention for each head
        """

        # Step 1: Extract dimensions
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Input dimension mismatch in MultiHeadAttention.forward().\n"
                f"  Expected: embed_dim={self.embed_dim} (set during initialization)\n"
                f"  Got: embed_dim={embed_dim} from input shape {x.shape}\n"
                f"  Fix: Ensure input tensor's last dimension matches the embed_dim used when creating MultiHeadAttention."
            )

        # Step 2: Project to Q, K, V
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # Step 3: Reshape to separate heads
        # From (batch, seq, embed_dim) to (batch, seq, num_heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Transpose to (batch, num_heads, seq, head_dim) for parallel processing
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 5: Apply attention
        # We can apply attention to all heads at once because scaled_dot_product_attention
        # supports broadcasting or 4D tensors if implemented correctly.

        # Reshape mask if necessary to broadcast over heads
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_len_mask, _ = mask.shape
            mask_data = mask.data.reshape(
                batch_size_mask, 1, seq_len_mask, seq_len_mask
            )
            mask_reshaped = Tensor(mask_data)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask_reshaped)

        # Step 6: Concatenate heads back together
        # Transpose back: (batch, num_heads, seq, head_dim) → (batch, seq, num_heads, head_dim)
        attended = attended.transpose(1, 2)

        # Reshape: (batch, seq, num_heads, head_dim) → (batch, seq, embed_dim)
        concat_output = attended.reshape(batch_size, seq_len, self.embed_dim)

        # Step 7: Apply output projection
        output = self.out_proj.forward(concat_output)

        return output

    def __call__(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Make MultiHeadAttention callable like attention(x)."""
        return self.forward(x, mask)

    def parameters(self) -> list[Tensor]:
        """
        Return all trainable parameters.

        Returns:
            List of all parameter tensors

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> params = mha.parameters()
        >>> print(len(params))  # 8 (4 layers × 2 params each: weight + bias)
        >>> print(params[0].shape)  # (64, 64) - q_proj weight
        >>> print(params[1].shape)  # (64,) - q_proj bias
        """
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
