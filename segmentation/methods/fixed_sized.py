import math
from dataclasses import dataclass

from parsing.model.parsing_result import ParsingBoundingBox, ParsingResult, ParsingResultType
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import Chunk, ChunkingResult
from segmentation.model.document_chunker import DocumentChunker


# TODO: Move dataclasses to protocol or own file
@dataclass
class RichToken:
    element_id: str
    token_index: int
    token: str


@dataclass
class ElementInfo:
    geom: list[ParsingBoundingBox]
    token_count: int

    @property
    def geom_count(self) -> int:
        return len(self.geom)

    @property
    def tokens_per_geom(self) -> float:
        # Assumes each bounding box contains the same amount of tokens
        # TODO: Change to depend on the height of the bounding box
        return self.token_count / self.geom_count


def _get_max_min_per_elem(element_ids: list[tuple]):
    """
    Finds the lowest and highest token indices for each element.

    Args:
        element_ids: List of Tuples of the structure: (``element_id``, ``token_index``)

    """
    elem_min_max = {}

    for elem_id, token_idx in element_ids:
        min_idx, max_idx = elem_min_max.get(elem_id, (token_idx, token_idx))

        if token_idx <= min_idx:
            elem_min_max[elem_id] = (token_idx, max_idx)
        if token_idx >= max_idx:
            elem_min_max[elem_id] = (min_idx, token_idx)

    return elem_min_max


class FixedSizeChunker(DocumentChunker):
    """Uses the fixed Size chunking strategy for document chunking."""

    max_tokens: int
    module = Chunkers.FIXED_SIZE

    excluded_types: list[ParsingResultType] = [
        ParsingResultType.TABLE,
        ParsingResultType.TABLE_ROW,
        ParsingResultType.FIGURE
    ]

    def __init__(self, options: dict = None):
        if options:
            self.max_tokens = options.get("max_tokens", 128)
            self.overlap = options.get("overlap", 32)
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

            elem_tokens_raw = self._encode(elem.content)

            # For each element we save the total token count and the bounding boxes
            token_cnt = len(elem_tokens_raw)
            elem_info[elem.id] = ElementInfo(elem.geom, token_cnt)

            # Save element id and the token index inside the
            elem_tokens = [
                RichToken(elem.id, idx, token)
                for idx, token
                in enumerate(elem_tokens_raw)
            ]

            # Add Tokens to the end of the token queue
            tokens.extend(elem_tokens)

            while len(tokens) > self.max_tokens:
                chunk_tokens = tokens[:self.max_tokens]

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

    # TODO: Move this into the Protocol
    def _get_chunk(self, buffer_slice: list[RichToken], idx: int,
                   elem_info: dict[str, ElementInfo]) -> Chunk:

        # List of tokenized text
        content = self._decode([t.token for t in buffer_slice])

        # Get the min and max token indices for each element that are included in the chunk
        elem_ids = [(t.element_id, t.token_index) for t in buffer_slice]
        elem_min_max = _get_max_min_per_elem(elem_ids)

        boxes = []

        for elem_id, min_max in elem_min_max.items():
            min_idx, max_idx = min_max
            info = elem_info[elem_id]

            if info.geom_count == 0: continue

            for ind, bbox in enumerate(info.geom):
                box_start_idx = ind * info.tokens_per_geom
                box_end_idx = box_start_idx + info.tokens_per_geom
                line_cnt = len(bbox.spans)

                # Handle no bbox tokens are in the chunk
                no_intersect = max_idx < box_start_idx or box_end_idx < min_idx
                if no_intersect: continue

                # Handle all bbox tokens are in the chunk
                full_intersect = min_idx <= box_start_idx and max_idx >= box_end_idx
                if full_intersect or line_cnt < 2:
                    boxes.append(bbox)
                    continue

                # Handle some bbox tokens are in the chunk
                copy_box = ParsingBoundingBox(
                    page=bbox.page,
                    left=bbox.left,
                    top=bbox.top,
                    right=bbox.right,
                    bottom=bbox.bottom,
                )

                # Assumes token density is the same across all lines
                token_per_line = info.tokens_per_geom / line_cnt

                if min_idx > box_start_idx:
                    frac_lines = (min_idx - box_start_idx) / token_per_line
                    top_line = math.floor(frac_lines)
                    copy_box.top = bbox.spans[top_line].top

                if max_idx < box_end_idx - 1:
                    frac_lines = (box_end_idx - max_idx) / token_per_line
                    bottom_line = line_cnt - math.ceil(frac_lines)
                    copy_box.bottom = bbox.spans[bottom_line].bottom

                boxes.append(copy_box)

        # Add relevant chunk metadata
        chunk_id = f"c_{idx}"
        token_cnt = len(buffer_slice)
        meta = {
            "token_len": token_cnt
        }

        return Chunk(id=chunk_id, content=content, metadata=meta, geom=boxes)
