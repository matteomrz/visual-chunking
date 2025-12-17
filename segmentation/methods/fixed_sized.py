from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import RichToken


class FixedSizeChunker(DocumentChunker):
    """Uses the fixed Size chunking strategy for document chunking."""

    module = Chunkers.FIXED_SIZE

    max_tokens: int
    overlap: int

    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.overlap = kwargs.get("overlap", 32)

    def _get_chunk_tokens(self, document: ParsingResult):
        tokens: list[RichToken] = []

        for elem in document.flatten():
            # Ignore unwanted types
            if elem.type in self.excluded_types:
                continue

            # Add the new tokens to the queue
            elem_tokens = self._encode(elem)
            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                # Always return fixed amount of tokens
                yield tokens[: self.max_tokens]

                cutoff = self.max_tokens - self.overlap
                tokens = tokens[cutoff:]

        # Residual Tokens form undersized chunk
        if tokens:
            yield tokens
