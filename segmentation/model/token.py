from dataclasses import dataclass

from parsing.model.parsing_result import ParsingBoundingBox


@dataclass
class RichToken:
    """Tokenized text piece with positional information."""

    element_id: str
    token_index: int
    token: int
    text: str


@dataclass
class ElementInfo:
    """Token and bounding box information for a page element."""

    geom: list[ParsingBoundingBox]
    token_count: int

    @property
    def geom_count(self) -> int:
        return len(self.geom)

    @property
    def tokens_per_geom(self) -> float:
        # Assumes each bounding box contains the same amount of tokens
        # TODO: Change to depend on the height of the bounding box
        return self.token_count / self.geom_count
