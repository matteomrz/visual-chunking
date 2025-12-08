from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker, get_chunk
from segmentation.model.token import ElementInfo, RichToken


class RecursiveChunker(DocumentChunker):
    module = Chunkers.RECURSIVE

    markers: list[str] = [
        "\n\n", "\n", ".", ";", ",", " ", ""
    ]

    def __init__(self, options: dict = None):
        if options:
            self.max_tokens = options.get("max_tokens", 128)
        else:
            self.max_tokens = 128

    def _segment(self, document: ParsingResult, options: dict = None) -> ChunkingResult:
        result = ChunkingResult(metadata=document.metadata)

        chunk_idx = 0
        tokens: list[RichToken] = []  # Token Queue
        elem_info: dict[str, ElementInfo] = {}

        for elem in document.flatten():

            if elem.type in self.excluded_types:
                continue

            elem_tokens, info = self._encode(elem)
            elem_info[elem.id] = info

            tokens.extend(elem_tokens)

            if len(tokens) > self.max_tokens:
                splits = self._find_split(tokens, 0)
                splits.pop()  # Last split often really undersized, so reuse it

                prev_split = 0
                for split in splits:
                    token_slice = tokens[prev_split:split]

                    chunk = get_chunk(token_slice, chunk_idx, elem_info)
                    result.chunks.append(chunk)
                    chunk_idx += 1

                    prev_split = split

                tokens = tokens[splits[-1]:]

        if len(tokens) > 0:
            chunk = get_chunk(tokens, chunk_idx, elem_info)
            result.chunks.append(chunk)

        return result

    def _find_split(self, tokens: list[RichToken], level: int, start_idx: int = 0) -> list[int]:
        split_char = self.markers[level]
        splits: list[int] = []

        for idx, token in enumerate(tokens):
            if split_char in token.text:
                splits.append(idx + 1)

        splits.append(len(tokens))

        rec_splits = []
        prev_split = 0
        prev_distance = 0
        for split in splits:
            distance = split - prev_split
            if distance > self.max_tokens:
                token_slice = tokens[prev_split:split]
                lower = self._find_split(token_slice, level + 1, prev_split)
                rec_splits.extend(lower)
                prev_distance = lower[1] - lower[0]
            else:
                can_merge = distance + prev_distance <= self.max_tokens
                if prev_split > 0 and can_merge:
                    rec_splits.pop()
                    distance += prev_distance

                rec_splits.append(start_idx + split)
                prev_distance = distance
            prev_split = split

        return rec_splits
