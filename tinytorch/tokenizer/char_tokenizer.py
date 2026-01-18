from .tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that treats each character as a separate token.

    This is the simplest tokenization approach - every character in the
    vocabulary gets its own unique ID.
    """

    def __init__(self, vocab: list[str] | None = None):
        """
        Initialize character tokenizer.

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.vocab_size
        4  # 3 chars + 1 unknown token
        """
        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = ["<UNK>"] + vocab
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = dict(enumerate(self.vocab))

        # Store unknown token ID
        self.unk_id = 0

    def build_vocab(self, corpus: list[str]) -> None:
        """
        Build vocabulary from a corpus of text.
        """
        # Collect all unique characters
        all_chars: set[str] = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(all_chars)

        # Rebuild vocabulary with <UNK> token first
        self.vocab = ["<UNK>"] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = dict(enumerate(self.vocab))

    def encode(self, text: str) -> list[int]:
        """
        Encode text to list of character IDs.

        APPROACH:
        1. Iterate through each character in text
        2. Look up character ID in vocabulary
        3. Use unknown token ID for unseen characters

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.encode("hello")
        [1, 2, 3, 3, 4]  # maps to h,e,l,l,o
        """

        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode list of token IDs back to text.

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.decode([1, 2, 3, 3, 4])
        "hello"
        """

        chars = []
        for token in tokens:
            chars.append(self.id_to_char.get(token, "<UNK>"))

        return "".join(chars)
