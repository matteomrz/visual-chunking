from lib.parsing.model.parsing_result import ParsingResult
from lib.segmentation.methods.chunkers import Chunkers
from lib.segmentation.model.document_chunker import DocumentChunker
from lib.segmentation.model.token import RichToken


class FixedSizeChunker(DocumentChunker):
    """Uses the fixed Size chunking strategy for document chunking."""

    module = Chunkers.FIXED_SIZE

    overlap: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overlap = kwargs.get("overlap", 32)

    def _get_chunk_tokens(self, document: ParsingResult):
        tokens: list[RichToken] = []

        for elem in document.flatten():
            # Ignore unwanted types
            if elem.type in self.excluded_types:
                continue

            # Add the new tokens to the queue
            elem_tokens = self._tokenize(elem)
            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                # Always return fixed amount of tokens
                yield tokens[: self.max_tokens]

                cutoff = self.max_tokens - self.overlap
                tokens = tokens[cutoff:]

        # Residual Tokens form undersized chunk
        if tokens:
            yield tokens
