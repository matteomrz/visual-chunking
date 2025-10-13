from pathlib import Path

# Collection of Path Objects to be used for file access

ROOT_DIR = Path(__file__).parent
PARSING_DIR = ROOT_DIR / "parsing"
GUIDELINES_DIR = PARSING_DIR / "guidelines"
BOUNDING_BOX_DIR = PARSING_DIR / "bounding-boxes"
ANNOTATED_DIR = PARSING_DIR / "annotated"

# Default File and Modules used by the parsing and drawing methods

DEFAULT_GUIDELINE = "example-guideline"
DEFAULT_MODULE = "unstructured-io"
