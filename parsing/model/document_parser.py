import json
from pathlib import Path
from typing import Protocol, TypeVar

from config import BOUNDING_BOX_DIR, GUIDELINES_DIR
from parsing.model.options import ParserOptions
from parsing.scripts.annotate import create_annotation
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

    @property
    def image_path(self) -> Path:
        return BOUNDING_BOX_DIR / self.module.value / "images"

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

    def _save(self, file_name: str, result: ParsingResult, batch_name: str = None):
        """
        Saves the transformed ParsingResult as a JSON file.

        The file will be saved in the parser's `dst_path` directory
        with the name `<file_name>-output.json`.

        Args:
            file_name: The base name of the original file
            result: The ParsingResult object to serialize and save
            batch_name: For Batch processing, name of the batch that the file belongs to [optional]
        """

        output_dir = self.dst_path
        if batch_name is not None:
            output_dir = output_dir / batch_name

        output_path = output_dir / f"{file_name}.json"

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)
            print(f"Success: JSON saved at: {output_path}")

    def _annotate(self, file_name: str, options: dict = None, batch_name: str = None):
        """
        Call drawing method if requested.
        To request drawing, set options["draw"] = True

        Args:
            file_name: The base name of the original file
            options: Dictionary
            batch_name: For Batch processing, name of the batch that the file belongs to [optional]
        """
        if options and options.get(ParserOptions.ANNOTATE, False):
            print(f"Scheduled annotation of {file_name}...")
            src_name = file_name
            if batch_name is not None:
                src_name = f"{batch_name}/{src_name}"

            create_annotation(parser_name=self.module.value, src_name=src_name, is_batch=False)

    def process_document(self, file_path: Path, batch_name: str = None, options: dict = None):
        """
        Performs full parsing pipeline for a single document.

        Args:
            file_path: Path of the guideline PDF file
            batch_name: For Batch processing, name of the batch that the file belongs to [optional]
            options: A dictionary of method-specific options [optional]

        Raises:
            FileNotFoundError: If there is no PDF file at the specified path
        """

        file_name = file_path.name
        if not (file_path.exists() and file_name and file_name.endswith(".pdf")):
            raise FileNotFoundError(f"Error: PDF not found: {file_path}")

        file_name = file_path.stem
        print(f"Parsing {file_name} using {self.module.name}...")

        json_result = self._parse(file_path, options)
        transformed_result = self._transform(json_result)

        self._save(file_name, transformed_result, batch_name=batch_name)
        self._annotate(file_name, options, batch_name=batch_name)

    def process_batch(self, batch_name: str, options: dict = None):
        """
                Performs full parsing pipeline for a batch of multiple documents.

                Args:
                    batch_name: Name of the directory containing the guideline PDF files in
                    options: A dictionary of method-specific options [optional]

                Raises:
                    FileNotFoundError: If the PDF file (`{file_name}.pdf`)
                                       is not found in `src_path`.
                """
        batch_path = self.src_path / batch_name

        if batch_path.exists() and batch_path.is_dir():
            for file_path in batch_path.glob("*.pdf"):
                self.process_document(file_path, batch_name, options)
        else:
            raise ValueError(f"Error: Path {batch_path} does not exist or is not a directory.")
