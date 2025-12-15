from typing import Optional

from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import ElementInfo, RichToken


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

    def segment(self, document: ParsingResult, with_geom: bool = True) -> ChunkingResult:
        result = ChunkingResult(metadata=document.metadata)
        document.add_delimiters()

        chunk_idx = 0
        tokens: list[RichToken] = []  # Token Queue
        elem_info: dict[str, ElementInfo] = {}

        for elem in document.flatten():
            if elem.type in self.excluded_types:
                continue

            # Get tokens and element info
            elem_tokens, info = self._encode(elem)
            elem_info[elem.id] = info

            # Add tokens to the end of the token queue
            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                chunk_tokens = tokens[: self.max_tokens]

                chunk = self.get_chunk(chunk_tokens, chunk_idx, elem_info, with_geom)
                result.chunks.append(chunk)
                chunk_idx += 1

                # Leave overlap tokens in the token queue
                tokens = tokens[self.max_tokens - self.overlap:]

        # Handle trailing undersized chunk
        if tokens:
            chunk = self.get_chunk(tokens, chunk_idx, elem_info, with_geom)
            result.chunks.append(chunk)

        return result
