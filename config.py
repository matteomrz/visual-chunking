from pathlib import Path

# Collection of Path Objects to be used for file access

ROOT_DIR = Path(__file__).parent

THESIS_DIR = ROOT_DIR / "thesis"  # Latex Thesis directory
FIGURE_DIR = THESIS_DIR / "figures"  # Tex files for all figures
TABLE_DIR = FIGURE_DIR / "tables"  # Tex files for tables
PLOT_DIR = FIGURE_DIR / "plots"  # csv files containing data for plotting

LIB_DIR = ROOT_DIR / "lib"  # Code directory

PARSING_DIR = LIB_DIR / "parsing"  # Code for Parsing Module
CHUNKING_DIR = LIB_DIR / "chunking"  # Code for Chunking Module
EVAL_DIR = LIB_DIR / "evaluation"  # Code for Evaluating the Parsing and Segmentation Modules

DATA_DIR = ROOT_DIR / "data"  # Contains all input and output files
GUIDELINES_DIR = DATA_DIR / "guidelines"  # Input Guideline PDF files
PARSING_RESULT_DIR = DATA_DIR / "parsing-result"  # ParsingResult as JSON
ANNOTATED_DIR = DATA_DIR / "annotated"  # Guideline PDF file with annotated bounding boxes
IMAGES_DIR = DATA_DIR / "images"  # Images produced by the parsing method
MD_DIR = DATA_DIR / "markdown"  # Parsing output as MD
CHUNKING_RESULT_DIR = DATA_DIR / "chunking-result"  # ChunkingResult as JSON
CONFIG_DIR = DATA_DIR / "configs"  # Configs for benchmarks

# Path pointing to the OmniDocBench project

OMNI_DOC_PROJECT_PATH = ROOT_DIR.parent / "omni_doc" / "OmniDocBench"
