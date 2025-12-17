from dataclasses import dataclass


@dataclass
class RichToken:
    """Tokenized text piece with positional information."""

    element_id: str
    token_index: int
    token: int
    text: str
