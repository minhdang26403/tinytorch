from tinytorch.tokenizer import Tokenizer


def test_unit_base_tokenizer():
    """🔬 Test base tokenizer interface."""
    print("🔬 Unit Test: Base Tokenizer Interface...")

    # Test that base class defines the interface
    tokenizer = Tokenizer()

    # Should raise NotImplementedError for both methods
    try:
        tokenizer.encode("test")
        assert False, "encode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        tokenizer.decode([1, 2, 3])
        assert False, "decode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    print("✅ Base tokenizer interface works correctly!")
