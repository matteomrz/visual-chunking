import json
import random

import argparse
from pymupdf import Document, pymupdf

from config import ANNOTATED_DIR, BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, GUIDELINES_DIR
from parsing.methods.config import Parsers

color_mapping: dict[str, tuple[float, float, float]] = {}

DEFAULT_PARSER = Parsers.default().value
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=DEFAULT_GUIDELINE,
    help=f'PDF filename without extension. Default: "{DEFAULT_GUIDELINE}"',
)
arg_parser.add_argument(
    "--parser",
    "-p",
    type=str,
    default=DEFAULT_PARSER,
    help=f'Supported PDF parsing method. Default: "{DEFAULT_PARSER}"',
    choices=[p.value for p in Parsers],
)


def _get_color(label: str):
    if not color_mapping.get(label):
        color_mapping[label] = (random.random(), random.random(), random.random())
    return color_mapping[label]


def _draw_element(element: dict, doc: Document):
    loaded_idx = -1
    page = None

    for box in element.get("geom", []):
        page_idx = box.get("page", 1) - 1

        if page_idx < 0 or page_idx >= doc.page_count:
            print(f"Warning: Malformed page number [{page_idx + 1}] in {element}")
            continue

        if loaded_idx != page_idx:
            page = doc.load_page(page_idx)

        page_size = page.rect
        page_height = page_size.height
        page_width = page_size.width

        x1 = box.get("x", 0.0) * page_width
        y1 = box.get("y", 0.0) * page_height
        x2 = x1 + box.get("w", 0.0) * page_width
        y2 = y1 + box.get("h", 0.0) * page_height

        label = element.get("type", "undefined")
        rect = pymupdf.Rect(x1, y1, x2, y2)
        color = _get_color(label)

        page.draw_rect(rect=rect, color=color, width=1.5)
        page.insert_text(point=(x1, y1 - 10), text=label, fontsize=8, color=color, fill=color, fill_opacity=0.3)

    for child in element.get("children", []):
        if isinstance(child, dict):
            _draw_element(child, doc)
        else:
            print(f"Warning invalid child element of type: {type(child)}")


def draw_annotations(file_name: str, parser: Parsers):
    json_path = BOUNDING_BOX_DIR / parser.value / f"{file_name}-output.json"
    doc_path = GUIDELINES_DIR / f"{file_name}.pdf"
    output_dir = ANNOTATED_DIR / parser.value
    output_path = output_dir / f"{file_name}-annotated.pdf"

    if not doc_path.exists():
        raise ValueError(f"File not found: {doc_path}")

    if not json_path.exists():
        raise ValueError(f"File not found: {json_path}")

    with open(json_path) as j:
        annotations = json.load(j)
        document = pymupdf.open(doc_path)

        if isinstance(annotations, dict):
            for element in annotations.get("children", []):
                if isinstance(element, dict):
                    _draw_element(element, document)
                else:
                    print(f"Warning invalid child element of type: {type(element)}")

        output_dir.mkdir(parents=True, exist_ok=True)
        document.save(output_path)


def _draw():
    parser = Parsers.get_parser(args.parser)
    draw_annotations(file_name=args.file, parser=parser)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    _draw()
