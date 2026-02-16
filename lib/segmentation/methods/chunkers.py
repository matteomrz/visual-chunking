from enum import Enum


class Chunkers(Enum):
    """Available methods for segmenting ParsingResult into Chunks."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"

    @classmethod
    def get_chunker_type(cls, name: str):
        return cls(name)
