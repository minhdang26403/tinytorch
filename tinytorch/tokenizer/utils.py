import numpy as np

from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharTokenizer
from .tokenizer import Tokenizer


def create_tokenizer(
    strategy: str = "char", vocab_size: int = 1000, corpus: list[str] | None = None
) -> Tokenizer:
    """
    Factory function to create and train tokenizers.

    APPROACH:
    1. Check strategy type
    2. Create appropriate tokenizer class
    3. Train on corpus if provided
    4. Return configured tokenizer

    EXAMPLE:
    >>> corpus = ["hello world", "test text"]
    >>> tokenizer = create_tokenizer("char", corpus=corpus)
    >>> tokens = tokenizer.encode("hello")
    """
    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'.\n"
            f"  Available strategies: 'char', 'bpe'.\n"
            f"  Fix: Use 'char' for character-level or 'bpe' for byte-pair encoding tokenization."
        )

    return tokenizer


def tokenize_dataset(
    texts: list[str], tokenizer: Tokenizer, max_length: int | None = None
) -> list[list[int]]:
    """
    Tokenize a dataset with optional length limits.

    APPROACH:
    1. Encode each text with the tokenizer
    2. Apply max_length truncation if specified
    3. Return list of tokenized sequences

    EXAMPLE:
    >>> texts = ["hello world", "tokenize this"]
    >>> tokenizer = CharTokenizer(['h','e','l','o',' ','w','r','d','t','k','n','i','z','s'])
    >>> tokenized = tokenize_dataset(texts, tokenizer, max_length=10)
    >>> all(len(seq) <= 10 for seq in tokenized)
    True

    HINTS:
    - Handle empty texts gracefully (empty list is fine)
    - Truncate from the end if too long: tokens[:max_length]
    """
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized


def analyze_tokenization(texts: list[str], tokenizer: Tokenizer) -> dict[str, float]:
    """
    Analyze tokenization statistics.

    APPROACH:
    1. Tokenize all texts
    2. Compute sequence length statistics
    3. Calculate compression ratio
    4. Return analysis dictionary

    EXAMPLE:
    >>> texts = ["hello", "world"]
    >>> tokenizer = CharTokenizer(['h','e','l','o','w','r','d'])
    >>> stats = analyze_tokenization(texts, tokenizer)
    >>> 'vocab_size' in stats and 'avg_sequence_length' in stats
    True

    HINTS:
    - Use np.mean() for average sequence length
    - Compression ratio = total_characters / total_tokens
    - Return dict with vocab_size, avg_sequence_length, max_sequence_length, etc.
    """
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        "vocab_size": tokenizer.vocab_size,
        "avg_sequence_length": np.mean(tokenized_lengths),
        "max_sequence_length": max(tokenized_lengths) if tokenized_lengths else 0,
        "total_tokens": len(all_tokens),
        "compression_ratio": total_chars / len(all_tokens) if all_tokens else 0,
        "unique_tokens": len(set(all_tokens)),
    }

    return stats
