import json
import logging
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator

import numpy as np
from transformers import AutoTokenizer

from config import PARSING_RESULT_DIR, CHUNKING_RESULT_DIR
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingResult, ParsingResultType
from lib.chunking.methods.chunkers import Chunkers
from lib.chunking.model.chunk import Chunk, ChunkingResult
from lib.chunking.model.token import RichToken
from lib.utils.annotate import create_annotation
from lib.utils.create_dir import create_directory
from lib.utils.max_min import get_max_min
from lib.utils.open import open_parsing_result

logger = logging.getLogger(__name__)


def get_chunk(
    buffer_slice: list[RichToken],
    idx: int,
    elements: dict[str, ParsingResult],
    with_geom: bool = True,
) -> Chunk:
    """
    Creates a Chunk object and adds additional metadata.
    Handles the creation of partial bounding boxes for cases where Document Elements are split up between different chunks.

    Args:
        buffer_slice: List of RichToken, containing information about token content and position
        idx: Index of the resulting Chunk, used for setting the Chunk's ``id`` field
        elements: Mapping from ParsingResult id to ParsingResult
        with_geom: Whether to extract a bounding box for the resulting Chunk (Default: True)

    Returns:
        Chunk object
    """

    boxes = []

    if with_geom:
        # Get the min and max token indices for each element that are included in the chunk
        elem_ids = [(t.element_id, t.token_index) for t in buffer_slice]
        elem_min_max = get_max_min(elem_ids)

        for elem_id, min_max in elem_min_max.items():
            min_idx, max_idx = min_max
            elem = elements[elem_id]

            if elem.geom_count == 0:
                continue

            for ind, bbox in enumerate(elem.geom):
                box_start_idx = ind * elem.tokens_per_geom
                box_end_idx = box_start_idx + elem.tokens_per_geom
                line_cnt = len(bbox.spans)

                # Handle no bbox tokens are in the chunk
                no_intersect = max_idx < box_start_idx or box_end_idx < min_idx
                if no_intersect:
                    continue

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
                token_per_line = elem.tokens_per_geom / line_cnt

                # While in MOST cases just taking the floor / ceil of the frac_lines
                # works out to include the entire text, we prefer having increased
                # recall at the cost of being less precise

                # If you want to have closer bounding boxes remove the -1 / +1 after rounding

                if min_idx > box_start_idx:
                    frac_lines = (min_idx - box_start_idx) / token_per_line
                    top_line = max(0, math.floor(frac_lines) - 1)
                    copy_box.top = bbox.spans[top_line].top

                if max_idx < box_end_idx:
                    frac_lines = (box_end_idx - max_idx) / token_per_line
                    bottom_line = min(line_cnt - 1, line_cnt - math.ceil(frac_lines) + 1)
                    copy_box.bottom = bbox.spans[bottom_line].bottom

                boxes.append(copy_box)

    # Add relevant chunk metadata
    chunk_id = f"c_{idx}"
    content = "".join([t.text for t in buffer_slice]).strip()
    token_cnt = len(buffer_slice)
    meta = {
        "token_len": token_cnt
    }

    return Chunk(id=chunk_id, content=content, metadata=meta, geom=boxes)


class DocumentChunker(ABC):
    """
    Standard Interface for a Document Chunker.
    Transforms a ParsingResult into a ChunkingResult.
    """

    module: Chunkers

    src_path: Path = PARSING_RESULT_DIR

    # Exclude Types with empty content, if any, their children will supply the content
    excluded_types: list[ParsingResultType] = [
        ParsingResultType.TABLE_ROW,
    ]

    # Placeholder for now
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    max_tokens: int

    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 128)

    def _tokenize(self, element: ParsingResult) -> list[RichToken]:
        """
        Transforms a ParsingResults into a list of Tokens.
        Additionally, adds the total token count of the element to the element's metadata.

        Args:
            element: ParsingResult to be tokenized

        Returns:
            list[RichToken]: Tokens including metadata about their position in the element.
        """
        data = self.tokenizer(element.content, return_offsets_mapping=True)
        encoded = data.get("input_ids", [])
        offsets = data.get("offset_mapping", [])

        tokens = []
        text_start = 0

        drop_cnt = 0
        for idx in range(len(encoded)):
            # Avoid dropping white-space at the end of the element -> Important for Chroma evaluation
            if idx == len(encoded) - 1:
                text_end = len(element.content)
            else:
                text_end = offsets[idx][1]

            if text_end <= text_start:
                drop_cnt += 1
                continue

            code = encoded[idx]
            text = element.content[text_start: text_end]
            token = RichToken(element.id, idx - drop_cnt, code, text)
            tokens.append(token)

            text_start = text_end

        token_cnt = len(tokens)
        element.metadata["token_cnt"] = token_cnt

        return tokens

    def _get_token_count(self, text: str):
        return len(self.tokenizer(text)["input_ids"])

    def _decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def dst_path(self) -> Path:
        """
        Directory to save the ChunkingResult JSON serialization in.
        Example: 'SEGMENTATION_OUTPUT_DIR/fixed_size/1024_200/'
        """
        return CHUNKING_RESULT_DIR / self.module.value / self._config_str

    @property
    def _config_str(self):
        """
        Subdirectory name to differentiate between different chunking configurations.
        Example: '1024_200', for a chunker with max_token_length=1024 and overlap=200
        """
        return "_".join([str(v) for k, v in self._parameters.items()])

    @property
    def _parameters(self) -> dict[str, str | int]:
        """All set-able parameters for the chunking strategy."""
        return {
            k: v for k, v in vars(self).items()
            if isinstance(v, (str, int))
        }

    @abstractmethod
    def _get_chunk_tokens(self, document: ParsingResult) -> Generator[list[RichToken], Any, None]:
        """
        Splits up a document into lists of RichTokens.
        Each list represents the tokens that make up each chunk.

        Args:
            document: The input ParsingResult

        Yields:
            A list of RichTokens which should make up the next chunk.
        """

    def segment(self, document: ParsingResult, with_geom: bool = True) -> ChunkingResult:
        """
        Segments the ParsingResult at the given file path

        Args:
            document: The input ParsingResult
            with_geom: Whether to extract a bounding box for the resulting Chunks (Default: True)

        Returns:
            ChunkingResult containing a list of the created chunks
        """
        result = ChunkingResult(metadata=document.metadata)
        document.add_delimiters()

        # Allows for quick accessing of elements during chunk creation
        elem_info = {
            elem.id: elem
            for elem in document.flatten()
        }

        chunk_idx = 0
        for segment in self._get_chunk_tokens(document):
            if not isinstance(segment, list):
                break

            # Filter out any empty chunks (fix for recursive strategy)
            if not any([t.text.strip() for t in segment]):
                continue

            chunk = get_chunk(segment, chunk_idx, elem_info, with_geom)
            result.chunks.append(chunk)

            chunk_idx += 1

        return result

    def _save(self, file_path: Path, result: ChunkingResult):
        """
        Saves the created Chunks as a JSON file.

        The file will be saved to the chunker's `dst_path` directory
        with the name `<file_name>.json`.

        Args:
            file_path: The path of the bounding box file
            result: The ChunkingResult to be saved
        """
        output_dir = create_directory(file_path, self.src_path, self.dst_path)
        output_path = output_dir / file_path.name

        result.metadata["chunk_path"] = str(output_path)

        with open(output_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)

    def process_document(self, file_path: Path, with_geom: bool = True,
                         draw: bool = False) -> ChunkingResult:
        """
        Performs chunking for a single document.

        Args:
            file_path: Absolute path of the JSON file containing the bounding boxes
            with_geom: Whether to extract a bounding box for the resulting Chunks (Default: True)
            draw: Whether to create an annotated PDF file for the resulting Chunks (Default: False)

        Returns:
            Chunked document as ChunkingResult
        """
        document = open_parsing_result(file_path)
        file_name = file_path.stem
        logger.info(f"Chunking {file_name} using {self.module.name}...")

        start_time = time.time()
        result = self.segment(document, with_geom)

        chunk_time = time.time() - start_time
        self._add_metadata(result, chunk_time)
        self._save(file_path, result)

        if draw:
            create_annotation(result)

        logger.info(f"Finished chunking {file_name} in {chunk_time:.4f}s.")

        return result

    def process_batch(
        self,
        batch_name: str,
        with_geom: bool = True,
        draw: bool = False
    ) -> list[ChunkingResult]:
        """
        Performs chunking for a batch of multiple documents.

        Args:
            batch_name: Name of the directory containing the JSON files of the bounding boxes
            with_geom: Whether to extract a bounding box for the resulting Chunks (Default: True)
            draw: Whether to create annotated PDF files for the resulting Chunks (Default: False)

        Raises:
            FileNotFoundError: If the batch directory is not found in ``src_path``

        Returns:
            List of ChunkingResults for each file in the batch
        """
        batch_path = self.src_path / batch_name

        if batch_path.exists() and batch_path.is_dir():
            results = []

            for file_path in batch_path.glob("*.json"):
                res = self.process_document(file_path, with_geom, draw)
                results.append(res)

            return results
        else:
            raise ValueError(f"Error: {batch_path} does not exist or is not a directory.")

    def _add_metadata(self, result: ChunkingResult, chunk_time: float):
        """Add additional metadata to the result of the document chunking."""
        token_lens = [c.metadata["token_len"] for c in result.chunks]

        result.metadata.update(self._parameters)
        result.metadata["chunking_time"] = chunk_time
        result.metadata["chunk_count"] = len(result.chunks)
        result.metadata["chunk_length_mean"] = np.mean(token_lens).round(4)
        result.metadata["chunk_length_std"] = np.std(token_lens).round(4)
        result.metadata["chunk_length_median"] = np.median(token_lens)
