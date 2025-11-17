import json
import os
from pathlib import Path
from typing import Protocol, TypeVar

from config import BOUNDING_BOX_DIR, GUIDELINES_DIR
from parsing.draw import draw_bboxes
from parsing.methods.config import Parsers
from parsing.model.parsing_result import ParsingResult

# Raw output type of the PDF parsing method
T = TypeVar("T", default=dict)


class DocumentParser(Protocol[T]):
    """
    Standard Interface for a PDF Document Parser.
    Transforms a PDF file into a structured JSON format
    """

    module: Parsers  # Has to be set by the implementation
    src_path: Path = GUIDELINES_DIR

    @property
    def dst_path(self) -> Path:
        return BOUNDING_BOX_DIR / self.module.value

    def _parse(self, file_path: Path, options: dict = None) -> T:
        """
        Parses the document at the given file path.

        Args:
            file_path: The absolute path to the input PDF file
            options: parser-specific options [optional]

        Returns:
            Raw output from the parsing method
        """

    def _transform(self, raw_result: T) -> ParsingResult:
        """
        Transforms the raw parser output into a ParsingResult object.

        Args:
            raw_result: Raw output from the parsing method

        Returns:
            ParsingResult representation of the initial document
        """

    def _save(self, file_name: str, result: ParsingResult):
        """
        Saves the transformed ParsingResult as a JSON file.

        The file will be saved in the parser's `dst_path` directory
        with the name `<file_name>-output.json`.

        Args:
            file_name: The base name of the original file
            result: The ParsingResult object to serialize and save
        """
        output_path = self.dst_path / f"{file_name}-output.json"

        os.makedirs(self.dst_path, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)
            print(f"Success: JSON saved at: {output_path}")

    def _draw(self, file_name: str, options: dict):
        """
        Call drawing method if requested.
        To request drawing, set options["draw"] = True

        Args:
            file_name: The base name of the original file
            options: Dictionary
        """
        if options and options.get("draw", False):
            print("Drawing bounding boxes...")
            draw_bboxes(file_name, module_name=self.module)

    def process_document(self, file_name: str, options: dict = None):
        """
        Performs full parsing pipeline for a single document.

        Args:
            file_name: PDF file base name located in `src_path` directory
            options: A dictionary of method-specific options [optional]

        Raises:
            FileNotFoundError: If the PDF file (`{file_name}.pdf`)
                               is not found in `src_path`.
        """
        file_path = self.src_path / f"{file_name}.pdf"

        if not file_path.exists():
            raise FileNotFoundError(f"Error: PDF not found: {file_path}")

        print(f"Parsing {file_path.name} using {self.module.name}...")

        json_result = self._parse(file_path, options)
        transformed_result = self._transform(json_result)
        self._save(file_name, transformed_result)
