import json
from pathlib import Path

from parsing.model.parsing_result import ParsingResult


def open_parsing_result(file_path: Path) -> ParsingResult:
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
