from enum import Enum


class Chunkers(Enum):
    """Available methods for segmenting ParsingResult into Chunks."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
