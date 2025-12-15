from typing import Optional

from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import ElementInfo, RichToken

markers: list[str] = [
    "\n\n", "\n", ";", ".", ",", " ", ""
]
header_limiter = f":{markers[1]}"


class RecursiveChunker(DocumentChunker):
    """
    Uses the Recursive character chunking strategy for document chunking.
    Adapted from the Langchain implementation with an adapted list of separators.
    """

    module = Chunkers.RECURSIVE

    def __init__(self, max_tokens: Optional[int] = None, overlap: Optional[int] = None):
        self.max_tokens = max_tokens if max_tokens is not None else 128
        self.overlap = overlap if overlap is not None else 0

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

            elem_tokens, info = self._encode(elem)
            elem_info[elem.id] = info

            if len(elem_tokens) == 0:
                continue

            tokens.extend(elem_tokens)

            if len(tokens) > self.max_tokens:
                splits = find_rec_split(tokens, self.max_tokens)
                splits.pop()

                prev_split = 0
                for split in splits:
                    token_slice = tokens[prev_split:split]

                    chunk = self.get_chunk(token_slice, chunk_idx, elem_info, with_geom)
                    result.chunks.append(chunk)
                    chunk_idx += 1

                    prev_split = split

                cutoff = prev_split - self.overlap
                tokens = tokens[cutoff:]

        if len(tokens) > 0:
            chunk = self.get_chunk(tokens, chunk_idx, elem_info, with_geom)
            result.chunks.append(chunk)

        return result


def find_rec_split(
    tokens: list[RichToken],
    max_tokens: int,
    level: int = 0,
    start_idx: int = 0,
) -> list[int]:
    split_char = markers[level]
    splits: list[int] = []

    for idx, token in enumerate(tokens):
        if split_char in token.text:
            splits.append(idx + 1)

    max_idx = len(tokens)
    if max_idx not in splits:
        splits.append(len(tokens))

    rec_splits = []
    prev_split = 0
    prev_distance = 0
    for split in splits:
        distance = split - prev_split
        if distance > max_tokens:
            token_slice = tokens[prev_split:split]
            new_start_idx = start_idx + prev_split
            lower = find_rec_split(token_slice, max_tokens, level + 1, new_start_idx)
            rec_splits.extend(lower)
            prev_distance = lower[1] - lower[0]
        else:
            can_merge = distance + prev_distance <= max_tokens
            if prev_split > 0 and can_merge:
                rec_splits.pop()
                distance += prev_distance

            rec_splits.append(start_idx + split)
            prev_distance = distance
        prev_split = split

    return rec_splits
