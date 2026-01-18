from collections import Counter

from .tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that learns subword units.

    BPE works by:
    1. Starting with character-level vocabulary
    2. Finding most frequent character pairs
    3. Merging frequent pairs into single tokens
    4. Repeating until desired vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.

        APPROACH:
        1. Store target vocabulary size
        2. Initialize empty vocabulary and merge rules
        3. Set up mappings for encoding/decoding

        EXAMPLE:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> tokenizer.vocab_size
        1000
        """
        self.vocab_size = vocab_size
        self.vocab: list[str] = []
        self.merges: list[tuple[str, str]] = []  # List of (pair, new_token) merges
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _get_word_tokens(self, word: str) -> list[str]:
        """
        Convert word to list of characters with end-of-word marker.

        APPROACH:
        1. Split word into characters
        2. Add </w> marker to last character
        3. Return list of tokens

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o</w>']
        """
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += "</w>"  # Mark end of word
        return tokens

    def _get_pairs(self, word_tokens: list[str]) -> set[tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens.

        APPROACH:
        1. Iterate through adjacent tokens
        2. Create pairs of consecutive tokens
        3. Return set of unique pairs

        EXAMPLE:
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def train(self, corpus: list[str], vocab_size: int | None = None) -> None:
        """
        Train BPE on corpus to learn merge rules.

        APPROACH:
        1. Build initial character vocabulary
        2. Count word frequencies in corpus
        3. Iteratively merge most frequent pairs
        4. Build final vocabulary and mappings

        EXAMPLE:
        >>> corpus = ["hello", "hello", "help"]
        >>> tokenizer = BPETokenizer(vocab_size=20)
        >>> tokenizer.train(corpus)
        >>> len(tokenizer.vocab) <= 20
        True
        """
        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies
        word_freq = Counter(corpus)

        # Initialize vocabulary with characters
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        # Convert to sorted list for consistency
        self.vocab = sorted(vocab)

        # Add special tokens
        if "<UNK>" not in vocab:
            self.vocab = ["<UNK>"] + self.vocab

        # Learn merges
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            # Count all pairs across all words
            pair_counts: Counter[tuple[str, str]] = Counter()

            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                pairs = self._get_pairs(tokens)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Merge this pair in all words
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        # Merge pair
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        # Build final mappings
        self._build_mappings()

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = dict(enumerate(self.vocab))

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """
        Apply learned merge rules to token sequence.

        TODO: Apply BPE merges to token list

        APPROACH:
        1. Start with character-level tokens
        2. Apply each merge rule in order
        3. Continue until no more merges possible

        EXAMPLE:
        >>> # After training, merges might be [('h','e'), ('l','l')]
        >>> tokenizer._apply_merges(['h','e','l','l','o</w>'])
        ['he','ll','o</w>']  # Applied both merges

        HINT: For each merge pair, scan through tokens and replace adjacent pairs
        """
        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge_pair[0]
                    and tokens[i + 1] == merge_pair[1]
                ):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode text using BPE.

        APPROACH:
        1. Split text into words
        2. Convert each word to character tokens
        3. Apply BPE merges
        4. Convert to token IDs

        EXAMPLE:
        >>> tokenizer.encode("hello world")
        [12, 45, 78]  # Token IDs after BPE merging

        HINTS:
        - Use text.split() for simple word splitting
        - Use _get_word_tokens() to get character-level tokens for each word
        - Use _apply_merges() to apply learned merge rules
        - Use token_to_id dictionary with 0 (UNK) as default
        """
        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        TODO: Convert token IDs back to readable text

        APPROACH:
        1. Convert IDs to tokens
        2. Join tokens together
        3. Clean up word boundaries and markers

        EXAMPLE:
        >>> tokenizer.decode([12, 45, 78])
        "hello world"  # Reconstructed text

        HINTS:
        - Use id_to_token dictionary with '<UNK>' as default
        - Join all tokens into single string with ''.join()
        - Replace '</w>' markers with spaces for word boundaries
        """
        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, "<UNK>")
            token_strings.append(token)

        # Join and clean up
        text = "".join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace("</w>", " ")

        # Clean up extra spaces
        text = " ".join(text.split())

        return text
