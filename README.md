### Evaluating and Enhancing Location-Aware Visual Document Segmentation for Oncology Guidelines

This repository contains scripts to parse oncology guideline PDFs using different methods, normalize
their outputs into
a shared bounding-box schema, and optionally draw the detected regions back onto the original PDF
for visualization.

### Prerequisites

- **uv**: This project uses `uv` for dependency management and virtual environments.
- **Data locations**: All input and output files are stored under `data/` (see `config.py`):
    - `guidelines/`: Input PDF files
    - `bounding-boxes/`: JSON Parsing Output
    - `annotated/`: Annotated PDF files
- **Environment** (for LlamaParse only): Set `LLAMAPARSE_API_KEY` in a `.env` file at the repo root.

Setup with uv:

```bash
# Install dependencies and create the virtual environment
uv sync
```

### General usage pattern

All parsing and drawing workflows are orchestrated through `parsing/interface.py`. Invoke it via
`uv run`:

```bash
uv run python -m parsing.interface [options]
```

**Key flags:**

- `--help / -h`: Information on command usage
- `--file / -f`: Single PDF (without extension) in `data/guidelines/` to process.
- `--batch / -b`: Directory under `data/guidelines/` for batch processing.
- `--parser / -p`: Parser implementation. Defaults to `docling`.
- `--skip_existing / -S`: Skip items that already have outputs for the selected parser. (TODO)
- `--draw / -D`: Save annotated PDF file after parsing.
- `--only_draw`: Skip parsing; Only draw annotations using existing outputs.

**Example flows:**

```bash
# Parse the default guideline with the default parser
uv run python -m parsing.interface

# Parse a specific file with LlamaParse, then draw results
uv run python -m parsing.interface -f example-guideline -p llamaparse -d

# Batch process every PDF in data/guidelines/pubLayNet/ without drawing
uv run python -m parsing.interface -b pubLayNet
```

### Parsing result schema

Outputs are serialized as JSON representations of the `ParsingResult` dataclass. Each entry
contains:

- `id`: Stable identifier from the source parser.
- `type`: Either a `ParsingResultType` enum value or a parser-specific string.
- `content`: Extracted text for the element.
- `geom`: A list of `ParsingBoundingBox`.
- `image` (optional): Path to an extracted image representing figures or tables.
- `children` (optional): Nested `ParsingResult` structures when the element has subcomponents.
- `metadata` (optional): Additional parser metadata

Documents start with `ParsingResult.root()`, which carries parse-level metadata and contains the
full hierarchy via `children`.

### Troubleshooting

- Ensure your input file exists at `parsing/guidelines/FILE.pdf`.
- For LlamaParse, verify `.env` contains a valid `LLAMAPARSE_API_KEY`.
- If drawing fails, confirm the expected JSON path exists for the given `--module`.
