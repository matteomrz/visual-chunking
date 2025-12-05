from parsing.model.parsing_result import (
    ParsingResult,
    ParsingResultType,
)
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import ElementInfo, RichToken


class FixedSizeChunker(DocumentChunker):
    """Uses the fixed Size chunking strategy for document chunking."""

    max_tokens: int
    module = Chunkers.FIXED_SIZE

    excluded_types: list[ParsingResultType] = [
        ParsingResultType.TABLE,
        ParsingResultType.TABLE_ROW,
        ParsingResultType.FIGURE,
    ]

    def __init__(self, options: dict = None):
        if options:
            self.max_tokens = options.get("max_tokens", 128)
            self.overlap = options.get("overlap", 32)
        else:
            self.max_tokens = 128
            self.overlap = 32

    def _segment(self, document: ParsingResult, options: dict = None) -> ChunkingResult:
        result = ChunkingResult(metadata=document.metadata)

        chunk_idx = 0
        tokens: list[RichToken] = []  # Token Queue
        elem_info: dict[str, ElementInfo] = {}

        for elem in document.flatten():
            if elem.type in self.excluded_types:
                continue

            elem_tokens_raw = self._encode(elem.content)

            # For each element we save the total token count and the bounding boxes
            token_cnt = len(elem_tokens_raw)
            elem_info[elem.id] = ElementInfo(elem.geom, token_cnt)

            # Save element id and the token index inside the
            elem_tokens = [
                RichToken(elem.id, idx, token)
                for idx, token in enumerate(elem_tokens_raw)
            ]

            # Add Tokens to the end of the token queue
            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                chunk_tokens = tokens[: self.max_tokens]

                chunk = self._get_chunk(chunk_tokens, chunk_idx, elem_info)
                result.chunks.append(chunk)
                chunk_idx += 1

                # Leave overlap tokens in the token queue
                tokens = tokens[self.max_tokens - self.overlap:]

        # Handle trailing undersized chunk
        if tokens:
            chunk = self._get_chunk(tokens, chunk_idx, elem_info)
            result.chunks.append(chunk)

        return result
