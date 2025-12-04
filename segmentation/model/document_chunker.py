import json
import time
from pathlib import Path
from typing import Protocol

from transformers import AutoTokenizer

from config import BOUNDING_BOX_DIR, SEGMENTATION_OUTPUT_DIR
from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import ChunkingResult
from utils.create_dir import create_directory


def _open(file_path: Path) -> ParsingResult:
    """
    Opens the JSON file of the ParsingResult.
    Transforms it into a ParsingResult object.

    Args:
        file_path: The absolute path to the input ParsingResult JSON

    Returns:
        ParsingResult object

    Raises:
        KeyError, TypeError: If the JSON is malformed
        FileNotFoundError: If the JSON file (``{file_name}.json``)
                           is not found in ``src_path``
    """
    file_name = file_path.name
    if not (file_path.exists() and file_name and file_name.endswith(".json")):
        raise FileNotFoundError(f"Error: Bounding Boxes not found: {file_path}")

    with open(file_path, "r") as f:
        document = json.load(f)
        if not isinstance(document, dict):
            raise ValueError(f"Error: Not a valid JSON scheme at {file_path}")

        return ParsingResult.from_dict(document)


class DocumentChunker(Protocol):
    """
    Standard Interface for a Document Chunker.
    Transforms a ParsingResult into a ChunkingResult.
    """

    module: Chunkers

    src_path: Path = BOUNDING_BOX_DIR

    # Placeholder for now
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def _encode(self, text: str) -> list[str]:
        return self.tokenizer.encode(text)

    def _decode(self, tokens: list[str]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _get_token_count(self, text: str) -> int:
        return len(self._encode(text))

    @property
    def dst_path(self) -> Path:
        return SEGMENTATION_OUTPUT_DIR / self.module.value

    def _segment(self, document: ParsingResult, options: dict = None) -> ChunkingResult:
        """
        Segments the ParsingResult at the given file path

        Args:
            document: The input ParsingResult
            options: A dictionary of method-specific options [optional]

        Returns:
            ChunkingResult containing a list of the created chunks
        """

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

        with open(output_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)

    def process_document(self, file_path: Path, options: dict = None):
        """
        Performs chunking for a single document.

        Args:
            file_path: Absolute path of the JSON file containing the bounding boxes
            options: A dictionary of method-specific options [optional]
        """
        document = _open(file_path)
        file_name = file_path.stem
        print(f"Chunking {file_name} using {self.module.name}...")

        start_time = time.time()
        result = self._segment(document, options)

        chunk_time = time.time() - start_time
        _add_metadata(result, chunk_time)

        self._save(file_path, result)

    def process_batch(self, batch_name: str, options: dict = None):
        """
        Performs chunking for a batch of multiple documents.

        Args:
            batch_name: Name of the directory containing the JSON files of the bounding boxes
            options: A dictionary of method-specific options [optional]

        Raises:
            FileNotFoundError: If the batch directory is not found in ``src_path``
        """
        batch_path = self.src_path / batch_name

        if batch_path.exists() and batch_path.is_dir():
            for file_path in batch_path.glob("*.json"):
                self.process_document(file_path, options)
        else:
            raise ValueError(f"Error: {batch_path} does not exist or is not a directory.")


def _add_metadata(result: ChunkingResult, chunk_time: float):
    """Add additional metadata to the result of the document chunking."""
    result.metadata["chunking_time"] = chunk_time
