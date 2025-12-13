import json
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar
from tqdm import tqdm

from config import BOUNDING_BOX_DIR, GUIDELINES_DIR, IMAGES_DIR, MD_DIR
from parsing.model.options import ParserOptions
from parsing.scripts.postprocess import parse_post_process
from parsing.scripts.annotate import create_annotation
from parsing.methods.config import Parsers
from parsing.model.parsing_result import ParsingMetaData as PmD, ParsingResult, ParsingResultType
from utils.create_dir import create_directory, get_directory
from utils.open import open_parsing_result

# Raw output type of the PDF parsing method
T = TypeVar("T", default=dict)


def _set_time_meta(
    result: ParsingResult, start: float, parse: float, transformation: float
):
    """Set the parsing and transformation duration on the ParsingResult."""
    parsing_duration = parse - start
    transformation_duration = transformation - parse
    print(f"Parsing Time: {round(parsing_duration, 2)}s")
    result.metadata[PmD.PARSING_TIME.value] = parsing_duration
    result.metadata[PmD.TRANSFORMATION_TIME.value] = transformation_duration


class DocumentParser(Protocol[T]):
    """
    Standard Interface for a PDF Document Parser.
    Transforms a PDF file into a structured JSON format.
    """

    # Have to be set by the implementation
    module: Parsers
    label_mapping: dict[str, ParsingResultType]

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
        return create_directory(
            file_path,
            src_dir=self.src_path,
            dst_dir=self.base_image_dir,
            with_file=True,
        )

    def _parse(self, file_path: Path, options: dict = None) -> T:
        """
        Parses the document at the given file path.

        Args:
            file_path: The absolute path to the input PDF file
            options: parser-specific options [optional]

        Returns:
            Raw output from the parsing method
        """

    def _get_element_type(self, raw_type: str) -> ParsingResultType:
        """
        Transform a raw label from the classification of the method to ParsingResultType.
        Uses the label_mapping specified by the method.

        Args:
            raw_type: Label string from the output method

        Returns:
            Corresponding ParsingResultType
        """
        if raw_type not in self.label_mapping.keys():
            print(f"Warning: No mapping for label '{raw_type}' in {self.module}.")
            return ParsingResultType.MISSING

        return self.label_mapping[raw_type]

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

    def _get_json_output_path(self, file_path: Path) -> Path:
        """
        Get the Path, where the final

        Args:
            file_path: The path of the original file
        """
        output_dir = get_directory(file_path, self.src_path, self.json_dst_path)
        return output_dir / f"{file_path.stem}.json"

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

    def _annotate(self, file_path: Path, options: dict = None):
        """
        Call drawing method if requested.
        To request drawing, set options["draw"] = True

        Args:
            file_path: The path of the input PDF file
            options: A dictionary of method-specific options [optional]
        """
        if options and options.get(ParserOptions.ANNOTATE, False):
            print(f"Scheduled annotation of {file_path.name}...")

            create_annotation(src_path=file_path, parser=self.module)

    def _set_meta(
        self, result: ParsingResult, file_path: Path, start_time: float, parse_time: float,
        transformation_time: float
    ):
        """Set the metadata for the ParsingResult."""
        result.metadata[PmD.PARSER.value] = self.module.value
        result.metadata[PmD.GUIDELINE_PATH.value] = str(file_path)
        _set_time_meta(result, start_time, parse_time, transformation_time)

    def process_document(self, file_path: Path, options: dict = None) -> ParsingResult:
        """
        Performs full parsing pipeline for a single document.

        Args:
            file_path: Path of the guideline PDF file
            options: A dictionary of method-specific options [optional]

        Raises:
            FileNotFoundError: If there is no PDF file at the specified path

        Returns:
            Parsing Output as ParsingResult
        """

        file_name = file_path.name
        if not (file_path.exists() and file_name and file_name.endswith(".pdf")):
            raise FileNotFoundError(f"Error: PDF not found: {file_path}")

        file_name = file_path.stem
        print(f"Info: Parsing {file_name} using {self.module.name}...")

        start_time = time.time()

        raw_result = self._parse(file_path, options)
        print("Success: Parsing completed")

        parse_time = time.time()

        print("Info: Transforming output...")
        transformed_result = self._transform(raw_result)
        md_result = self._get_md(raw_result, file_path)
        print("Success: Transformation completed")

        transformation_time = time.time()
        self._set_meta(transformed_result, file_path, start_time, parse_time, transformation_time)
        parse_post_process(file_path, transformed_result)

        self._save_md(file_path, md_result)
        self._save_json(file_path, transformed_result)
        self._annotate(file_path, options)

        return transformed_result

    def process_batch(self, batch_name: str, options: dict[ParserOptions, Any] = None) -> list[
        ParsingResult]:
        """
        Performs full parsing pipeline for a batch of multiple documents.

        Args:
            batch_name: Name of the directory containing the guideline PDF files in
            options: A dictionary of method-specific options [optional]

        Raises:
            FileNotFoundError: If the PDF file (`{file_name}.pdf`)
                               is not found in `src_path`.
        Returns:
            List of parsing outputs for the documents in the batch as ParsingResult
        """
        batch_path = self.src_path / batch_name
        skip_existing = options.get(ParserOptions.EXIST_OK, False)

        if batch_path.exists() and batch_path.is_dir():
            results = []

            for file_path in tqdm(batch_path.glob("*.pdf")):
                output_path = self._get_json_output_path(file_path)
                if skip_existing and output_path.exists():
                    print(f"Skipping Document: {file_path.stem}. Output JSON already exists.")
                    res = open_parsing_result(output_path)
                    results.append(res)
                else:
                    res = self.process_document(file_path, options)
                    results.append(res)

            return results
        else:
            raise ValueError(f"Error: {batch_path} does not exist or is not a directory.")
