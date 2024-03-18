from dataclasses import dataclass

import tiktoken


@dataclass
class List:
    """Tokens of the tokenized text."""

    text: str
    encoder: tiktoken.Encoding
    tokens: list[int]

    def __len__(self) -> int:
        """Return the length of this."""
        return len(self.tokens)

    def read(self, n: int, offset: int = 0) -> str:
        """Read n tokens from offset and decode them."""
        return self.encoder.decode(self.tokens[offset : offset + n])

    @staticmethod
    def new(text: str, model_name: str) -> "List":
        """Return a new List of text for specified model."""
        encoder = tiktoken.encoding_for_model(model_name)
        tokens = encoder.encode(text)
        return List(text=text, encoder=encoder, tokens=tokens)
