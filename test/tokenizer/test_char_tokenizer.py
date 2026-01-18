from tinytorch.tokenizer import CharTokenizer


def test_unit_char_tokenizer():
    """🔬 Test character tokenizer implementation."""
    print("🔬 Unit Test: Character Tokenizer...")

    # Test basic functionality
    vocab = ["h", "e", "l", "o", " ", "w", "r", "d"]
    tokenizer = CharTokenizer(vocab)

    # Test vocabulary setup
    assert tokenizer.vocab_size == 9  # 8 chars + UNK
    assert tokenizer.vocab[0] == "<UNK>"
    assert "h" in tokenizer.char_to_id

    # Test encoding
    text = "hello"
    tokens = tokenizer.encode(text)
    expected = [1, 2, 3, 3, 4]  # h,e,l,l,o (based on actual vocab order)
    assert tokens == expected, f"Expected {expected}, got {tokens}"

    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    # Test unknown character handling
    tokens_with_unk = tokenizer.encode("hello!")
    assert tokens_with_unk[-1] == 0  # '!' should map to <UNK>

    # Test vocabulary building
    corpus = ["hello world", "test text"]
    tokenizer.build_vocab(corpus)
    assert "t" in tokenizer.char_to_id
    assert "x" in tokenizer.char_to_id

    print("✅ Character tokenizer works correctly!")
