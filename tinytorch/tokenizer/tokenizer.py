class Tokenizer:
    """
    Base tokenizer class providing the interface for all tokenizers.

    This defines the contract that all tokenizers must follow:
    - encode(): text → list of token IDs
    - decode(): list of token IDs → text
    """

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.

        APPROACH:
        1. Subclasses will override this method
        2. Return list of integer token IDs

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.encode("abc")
        [0, 1, 2]
        """
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(self, tokens: list[int]) -> str:
        """
        Convert list of token IDs back to text.

        APPROACH:
        1. Subclasses will override this method
        2. Return reconstructed text string

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.decode([0, 1, 2])
        "abc"
        """
        raise NotImplementedError("Subclasses must implement decode()")
