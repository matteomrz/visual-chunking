from pathlib import Path

# Collection of Path Objects to be used for file access

ROOT_DIR = Path(__file__).parent

THESIS_DIR = ROOT_DIR / "thesis"  # Latex Thesis directory
FIGURE_DIR = THESIS_DIR / "figures"  # Tex files for all figures
TABLE_DIR = FIGURE_DIR / "tables"  # Tex files for tables

LIB_DIR = ROOT_DIR / "lib"  # Code directory

PARSING_DIR = LIB_DIR / "parsing"  # Code for Parsing Module
SEGMENTATION_DIR = LIB_DIR / "segmentation"  # Code for Segmentation Module
EVAL_DIR = LIB_DIR / "evaluation"  # Code for Evaluating the Parsing and Segmentation Modules

DATA_DIR = ROOT_DIR / "data"  # Contains all input and output files
GUIDELINES_DIR = DATA_DIR / "guidelines"  # Input Guideline PDF files
BOUNDING_BOX_DIR = DATA_DIR / "bounding-boxes"  # ParsingResult as JSON
ANNOTATED_DIR = DATA_DIR / "annotated"  # Guideline PDF file with annotated bounding boxes
IMAGES_DIR = DATA_DIR / "images"  # Images produced by the parsing method
MD_DIR = DATA_DIR / "markdown"  # Parsing output as MD
SEGMENTATION_OUTPUT_DIR = DATA_DIR / "segmentation"  # ChunkingResult as JSON
CONFIG_DIR = DATA_DIR / "configs"  # Configs for benchmarks

# Default File used by the parsing and annotation methods

DEFAULT_GUIDELINE = "example_guideline"
