from typing import Optional

from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import RichToken


class FixedSizeChunker(DocumentChunker):
    """Uses the fixed Size chunking strategy for document chunking."""

    max_tokens: int
    module = Chunkers.FIXED_SIZE

    def __init__(self, max_tokens: Optional[int] = None, overlap: Optional[int] = None):
        self.max_tokens = max_tokens if max_tokens is not None else 128
        self.overlap = overlap if overlap is not None else 32

    @classmethod
    def from_options(cls, options: dict):
        return cls(
            max_tokens=options.get("max_tokens", None),
            overlap=options.get("overlap", None),
        )

    def _get_chunk_tokens(self, document: ParsingResult):
        tokens: list[RichToken] = []
        for elem in document.flatten():
            if elem.type in self.excluded_types:
                continue

            # Get tokens and element info
            elem_tokens = self._encode(elem)

            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                yield tokens[: self.max_tokens]
                tokens = tokens[self.max_tokens - self.overlap:]

        if tokens:
            yield tokens
