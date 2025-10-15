### Evaluating and Enhancing Location-Aware Visual Document Segmentation for Oncology Guidelines

This repository contains scripts to parse oncology guideline PDFs using different methods, normalize their outputs into
a shared bounding-box schema, and optionally draw the detected regions back onto the original PDF for visualization.

### Prerequisites

- **uv**: This project uses `uv` for dependency management and virtual environments.
- **Data locations**: Place your input PDFs in `parsing/guidelines/`. Outputs are written under`parsing/bounding-boxes/`
  and annotated PDFs under `parsing/annotated/`.
- **Environment** (for LlamaParse only): Set `LLAMAPARSE_API_KEY` in a `.env` file at the repo root.

Setup with uv:

```bash
# Install dependencies and create the virtual environment
uv sync

# Activate the environment for your shell session
source .venv/bin/activate
```

### General usage pattern

All commands are invoked with Python's module runner.

```bash
python -m parsing.METHOD_NAME.main [flags]
```

Replace `METHOD_NAME` with `llamaparse` or `unstructured-io`. See method-specific flags below.

### LlamaParse method

Run the LlamaParse pipeline and optionally draw the results.

```bash
python -m parsing.llamaparse.main [--file FILE] [--draw]
```

- **--file, -f**: PDF filename without extension. Defaults to the value of `DEFAULT_GUIDELINE` from `config.py`.
- **--draw, -d**: If set, draws the produced bounding boxes onto the original PDF and writes an annotated PDF to
  `parsing/annotated/llamaparse/`.

Notes:

- Requires `LLAMAPARSE_API_KEY` in your `.env`.
- Output JSON is saved to `parsing/bounding-boxes/llamaparse/FILE-output.json`.

Examples:

```bash
# Parse default guideline
python -m parsing.llamaparse.main

# Parse a specific file and create an annotated PDF
python -m parsing.llamaparse.main --file example-guideline --draw
```

### unstructured-io method

Run the unstructured.io partitioner with a selectable strategy and optionally draw the results.

```bash
python -m parsing.unstructured-io.main [--file FILE] [--strat STRATEGY] [--draw]
```

- **--file, -f**: PDF filename without extension. Defaults to `DEFAULT_GUIDELINE` from `config.py`.
- **--strat, -s**: Partitioning strategy. One of: `auto`, `hi_res` (default), `fast`, `ocr_only`.
- **--draw, -d**: If set, draws the produced bounding boxes onto the original PDF and writes an annotated PDF to
  `parsing/annotated/unstructured-io/`.

Notes:

- Images for `Image`/`Table` elements are exported to `parsing/bounding-boxes/unstructured-io/images/FILE/STRATEGY/`.
- Output JSON is saved to `parsing/bounding-boxes/unstructured-io/FILE-STRATEGY-output.json`.

Examples:

```bash
# Parse with default hi_res strategy
python -m parsing.unstructured-io.main

# Parse with fast strategy and draw
python -m parsing.unstructured-io.main --file example-guideline --strat fast --draw
```

### Drawing bounding boxes (standalone)

You can generate annotated PDFs from any existing normalized JSON using the drawing utility directly.

```bash
python -m parsing.draw --file FILE --module MODULE [--appendix APPENDIX]
```

- **--file, -f**: PDF filename without extension (must exist in `parsing/guidelines/`).
- **--module, -m**: Which module's output to draw. Typically `llamaparse` or `unstructured-io`.
- **--appendix, -a**: Optional suffix used in the JSON file name. Do not include the leading dash. For example, for
  `unstructured-io` with strategy `hi_res`, pass `--appendix hi_res` to match `FILE-hi_res-output.json`.

File naming rules the drawer expects:

- Without appendix: JSON at `parsing/bounding-boxes/MODULE/FILE-output.json`, annotated PDF at
  `parsing/annotated/MODULE/FILE-annotated.pdf`.
- With appendix: JSON at `parsing/bounding-boxes/MODULE/FILE-APPENDIX-output.json`, annotated PDF at
  `parsing/annotated/MODULE/FILE-APPENDIX-annotated.pdf`.

Examples:

```bash
# Draw for LlamaParse output
python -m parsing.draw --file example-guideline --module llamaparse

# Draw for unstructured-io output produced with strategy hi_res
python -m parsing.draw --file example-guideline --module unstructured-io --appendix hi_res
```

### Shared bounding-box schema

All method outputs are normalized to the same schema. This is the reference from `parsing/bounding-boxes/schema.json`:

```json
[
    {
        "content": "Example Text of the Chunk",
        "metadata": {
            "element_id": "Id of the Chunk - depends on the method",
            "type": "Type of the Chunk (Title, Heading, Text, Table...)",
            "layout": {
                "page_num": 1,
                "height": 1,
                "width": 1,
                "bbox": [
                    [
                        1.0,
                        1.0
                    ],
                    [
                        1.0,
                        1.0
                    ],
                    [
                        1.0,
                        1.0
                    ],
                    [
                        1.0,
                        1.0
                    ]
                ]
            },
            "image_path": "optional: If method allows extraction of tables and figures as images",
            "link_path": "optional: If method extracts link information"
        }
    }
]
```

### Troubleshooting

- Ensure your input file exists at `parsing/guidelines/FILE.pdf`.
- For LlamaParse, verify `.env` contains a valid `LLAMAPARSE_API_KEY`.
- If drawing fails, confirm the expected JSON path exists for the given `--module` and optional `--appendix`.
- If drawing fails with an Error from the PyPdf2 Package due to an unexpected format in the pdf file, the original pdf
  is likely corrupted. Open the file using a pdf viewer and use the 'Print to PDF' option on modern computers to
  generate a new file.
