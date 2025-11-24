import json
import time
from pathlib import Path
from typing import Protocol

from config import BOUNDING_BOX_DIR, CHUNK_DIR
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import Chunk
from utils.create_dir import create_directory


class DocumentChunker(Protocol):
    """
    Standard Interface for a Document Chunker.
    Transforms a ParsingResult into Chunks.
    """

    module: Chunkers

    src_path: Path = BOUNDING_BOX_DIR

    @property
    def dst_path(self) -> Path:
        return CHUNK_DIR / self.module.value

    def _segment(self, file_path: Path) -> list[Chunk]:
        """
        Segments the ParsingResult at the given file path

        Args:
            file_path: The absolute path to the input ParsingResult JSON

        Returns:
            List of Chunks
        """

    def _save(self, file_path: Path, chunks: list[Chunk]):
        """
        Saves the created Chunks as a JSON file.

        The file will be saved to the chunker's `dst_path` directory
        with the name `<file_name>.json`.

        Args:
            file_path: The path of the bounding box file
            chunks: The list of Chunks to be saved
        """
        output_dir = create_directory(file_path, self.src_path, self.dst_path)
        output_path = output_dir / file_path.name
        serialized = [c.to_json for c in chunks]
        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)

    def process_document(self, file_path: Path, options: dict = None):
        """"""
        file_name = file_path.name
        if not (file_path.exists() and file_name and file_name.endswith(".json")):
            raise FileNotFoundError(f"Error: Bounding Boxes not found: {file_path}")

        file_name = file_path.stem
        print(f"Chunking {file_name} using {self.module.name}...")

        start_time = time.time()
        chunks = self._segment(file_path)

        end_time = time.time()
        self._save(file_path, chunks)
