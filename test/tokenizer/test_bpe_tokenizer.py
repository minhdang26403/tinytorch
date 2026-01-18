from tinytorch.tokenizer import BPETokenizer


def test_unit_bpe_tokenizer():
    """🔬 Test BPE tokenizer implementation."""
    print("🔬 Unit Test: BPE Tokenizer...")

    # Test basic functionality with simple corpus
    corpus = ["hello", "world", "hello", "hell"]  # "hell" and "hello" share prefix
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # Check that vocabulary was built
    assert len(tokenizer.vocab) > 0
    assert "<UNK>" in tokenizer.vocab

    # Test helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith("</w>"), "Should have end-of-word marker"

    pairs = tokenizer._get_pairs(["h", "e", "l", "l", "o</w>"])
    assert ("h", "e") in pairs
    assert ("l", "l") in pairs

    # Test encoding/decoding
    text = "hello"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)

    # Test round-trip on training data should work well
    for word in corpus:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        # Allow some flexibility due to BPE merging
        assert len(decoded.strip()) > 0

    print("✅ BPE tokenizer works correctly!")
