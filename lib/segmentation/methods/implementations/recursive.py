from lib.parsing.model.parsing_result import ParsingResult
from lib.segmentation.methods.chunkers import Chunkers
from lib.segmentation.model.document_chunker import DocumentChunker
from lib.segmentation.model.token import RichToken

# Ordered list of delimiters
# Adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# Added punctuation for more meaningful splitting
delimiters: list[str] = [
    "\n\n", "\n", ".", ";", ",", " ", ""
]


class RecursiveChunker(DocumentChunker):
    """
    Uses the Recursive character chunking strategy for document chunking.
    Adapted from the Langchain implementation with an adapted list of separators.
    """

    module = Chunkers.RECURSIVE

    max_tokens: int
    overlap: int

    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.overlap = kwargs.get("overlap", 0)

    def _get_chunk_tokens(self, document: ParsingResult):
        tokens: list[RichToken] = []

        for elem in document.flatten():
            # Ignore unwanted types
            if elem.type in self.excluded_types:
                continue

            # Add the new tokens to the queue
            elem_tokens = self._encode(elem)
            tokens.extend(elem_tokens)

            if len(tokens) > self.max_tokens:
                # Get chunk breakpoints
                splits = find_splits(tokens, self.max_tokens)

                # Maybe last split can be merged with the first split of the next iteration
                # Therefore leave the last split in the queue
                splits.pop()

                prev_split = 0
                for split in splits:
                    # Return tokens as indicated by the splits
                    yield tokens[prev_split:split]
                    prev_split = split

                cutoff = max(0, prev_split - self.overlap)
                tokens = tokens[cutoff:]

        # Residual tokens form undersized chunk
        if tokens:
            yield tokens


def find_splits(
    tokens: list[RichToken],
    max_tokens: int,
    level: int = 0,
    start_idx: int = 0,
) -> list[int]:
    delimiter = delimiters[level]

    # Find tokens indicating possible Chunk breakpoints
    splits: list[int] = []

    for idx, token in enumerate(tokens):
        if delimiter in token.text:
            splits.append(idx + 1)

    # Add len(tokens) to simplify iteration over the split sections
    max_idx = len(tokens)
    if max_idx not in splits:
        splits.append(len(tokens))

    final_splits = []
    prev_split = 0
    prev_distance = 0

    for split in splits:
        distance = split - prev_split

        # The delimiter does not split the text fine enough
        if distance > max_tokens:
            split_tokens = tokens[prev_split:split]
            new_start_idx = start_idx + prev_split

            # Split up section further with the next delimiter
            recursive_splits = find_splits(
                split_tokens,
                max_tokens,
                level + 1,
                new_start_idx,
            )

            final_splits.extend(recursive_splits)

        # The split is a viable chunk
        else:
            can_merge = distance + prev_distance <= max_tokens

            # Merge undersized splits
            if prev_split > 0 and can_merge:
                final_splits.pop()
                distance += prev_distance

            final_splits.append(start_idx + split)

        prev_distance = distance
        prev_split = split

    return final_splits
