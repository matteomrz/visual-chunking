import json
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from tqdm import tqdm

from config import BOUNDING_BOX_DIR, GUIDELINES_DIR, IMAGES_DIR, MD_DIR
from parsing.model.options import ParserOptions
from parsing.scripts.annotate import create_annotation
from parsing.methods.config import Parsers
from parsing.model.parsing_result import ParsingResult
from utils.create_dir import create_directory, get_directory

# Raw output type of the PDF parsing method
T = TypeVar("T", default=dict)


class DocumentParser(Protocol[T]):
    """
    Standard Interface for a PDF Document Parser.
    Transforms a PDF file into a structured JSON format.
    """

    module: Parsers  # Has to be set by the implementation
    src_path: Path = GUIDELINES_DIR

    @property
    def json_dst_path(self) -> Path:
        return BOUNDING_BOX_DIR / self.module.value

    @property
    def md_dst_path(self) -> Path:
        return MD_DIR / self.module.value

    @property
    def base_image_dir(self) -> Path:
        return IMAGES_DIR / self.module.value

    def _create_image_dir(self, file_path: Path) -> Path:
        """Creates an image directory for the given file."""
        return create_directory(file_path, src_dir=self.src_path, dst_dir=self.base_image_dir,
                                with_file=True)

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

    def _get_md(self, raw_result: T, file_path: Path) -> str:
        """
            Get the model output in Markdown format.

            Args:
                raw_result: Raw output from the parsing method
                file_path: The path of the original file

            Returns:
                Markdown representation of the initial document
        """

    def _check_output_exists(self, file_path: Path) -> bool:
        """
        Checks if there already exists a JSON output for the given file.

        Args:
            file_path: The path of the original file
        """
        output_dir = get_directory(file_path, self.src_path, self.json_dst_path)
        output_path = output_dir / f"{file_path.stem}.json"
        return output_path.exists()

    def _save_md(self, file_path: Path, md: str):
        """
        Saves the output from the model as a Markdown file.

        The file will be saved to the parsers `md_dst_path` directory
        with the name `<file_name>.md`.

        Args:
            file_path: The path of the original file
            md: Markdown output as a string
        """
        output_dir = create_directory(file_path, self.src_path, self.md_dst_path)
        output_path = output_dir / f"{file_path.stem}.md"

        with open(output_path, "w") as f:
            f.write(md)
            print(f"Success: Model output saved at: {output_path}")

    def _save_json(self, file_path: Path, result: ParsingResult):
        """
        Saves the transformed ParsingResult as a JSON file.

        The file will be saved to the parser's `json_dst_path` directory
        with the name `<file_name>.json`.

        Args:
            file_path: The path of the original file
            result: The ParsingResult object to serialize and save
        """
        output_dir = create_directory(file_path, self.src_path, self.json_dst_path)
        output_path = output_dir / f"{file_path.stem}.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
            print(f"Success: Model output saved at: {output_path}")

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

        start_time = time.time()

        raw_result = self._parse(file_path, options)
        print("Success: Parsing completed")

        parse_time = time.time()

        print("Transforming output...")
        transformed_result = self._transform(raw_result)
        md_result = self._get_md(raw_result, file_path)

        transformation_time = time.time()
        _set_time_meta(transformed_result, start_time, parse_time, transformation_time)

        self._save_md(file_path, md_result)
        self._save_json(file_path, transformed_result)
        self._annotate(file_name, options, batch_name=batch_name)

    def process_batch(self, batch_name: str, options: dict[ParserOptions, Any] = None):
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
        skip_existing = options.get(ParserOptions.EXIST_OK, False)

        if batch_path.exists() and batch_path.is_dir():
            for file_path in tqdm(batch_path.glob("*.pdf")):
                if skip_existing and self._check_output_exists(file_path):
                    print(f"Skipping Document: {file_path.stem}. Output JSON already exists.")
                else:
                    self.process_document(file_path, batch_name, options)
        else:
            raise ValueError(f"Error: Path {batch_path} does not exist or is not a directory.")


def _set_time_meta(result: ParsingResult, start: float, parse: float, transformation: float):
    """Set the parsing and transformation duration on the ParsingResult."""
    parsing_duration = parse - start
    transformation_duration = transformation - parse
    print(f"Parsing Time: {round(parsing_duration, 2)}s")
    result.metadata["parsing_time"] = parsing_duration
    result.metadata["transformation_time"] = transformation_duration
