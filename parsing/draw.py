import json

import argparse
from pymupdf import Document, pymupdf

from config import ANNOTATED_DIR, BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, GUIDELINES_DIR
from parsing.methods.config import Parsers

# If two methods produce the same output with different labels, the colors will be identical
# 19 colors from Tab20 color scheme (from matplotlib)
possible_colors = [
    (0.1216, 0.4667, 0.7059), (0.6824, 0.7804, 0.9098), (1.0000, 0.4980, 0.0549),
    (1.0000, 0.7333, 0.4706), (0.1725, 0.6275, 0.1725), (0.5961, 0.8745, 0.5412),
    (0.8392, 0.1529, 0.1569), (1.0000, 0.5961, 0.5882), (0.5804, 0.4039, 0.7412),
    (0.7725, 0.6902, 0.8353), (0.5490, 0.3373, 0.2941), (0.7686, 0.6118, 0.5804),
    (0.8902, 0.4667, 0.7608), (0.9686, 0.7137, 0.8235), (0.4980, 0.4980, 0.4980),
    (0.7804, 0.7804, 0.7804), (0.7373, 0.7412, 0.1333), (0.8588, 0.8588, 0.5529),
    (0.0902, 0.7451, 0.8118)
]
possible_colors_count = len(possible_colors)
color_mapping: dict[str, tuple] = {}

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
        current_index = len(color_mapping) * 2 % possible_colors_count
        color_mapping[label] = possible_colors[current_index]

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
        page.insert_text(point=(x1, y1 - 3), text=label, fontsize=6, color=color, fill=color, fill_opacity=0.6)

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
