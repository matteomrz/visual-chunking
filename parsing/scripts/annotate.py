import json

from pathlib import Path

from pymupdf import Document, pymupdf

from config import ANNOTATED_DIR, BOUNDING_BOX_DIR, GUIDELINES_DIR

# If two methods produce the same output with different labels, the colors will still be identical
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

        l = box.get("l", 0.0) * page_width
        t = box.get("t", 0.0) * page_height
        r = box.get("r", 0.0) * page_width
        b = box.get("b", 0.0) * page_height

        label = element.get("type", "undefined")
        rect = pymupdf.Rect(l, t, r, b)
        color = _get_color(label)

        page.draw_rect(rect=rect, color=color, width=1.5)
        page.insert_text(point=(l, t - 3), text=label, fontsize=6, color=color, fill=color,
                         fill_opacity=0.6)

    for child in element.get("children", []):
        if isinstance(child, dict):
            _draw_element(child, doc)
        else:
            print(f"Warning invalid child element of type: {type(child)}")


def _annotate_file(json_path: Path, doc_path: Path) -> Document:
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
                    print(f"Warning: invalid child element of type: {type(element)}")
        else:
            print(
                f"Error: Invalid Parsing Output at: {json_path}. Expected: `dict`, Actual: `{type(annotations)}`")

        return document


def create_annotation(parser_name: str, src_name: str, is_batch: bool = False):
    parser_output = BOUNDING_BOX_DIR / parser_name
    parser_annotations = ANNOTATED_DIR / parser_name

    # Batch Logic
    if is_batch:
        batch_path = parser_output / src_name

        if batch_path.exists() and batch_path.is_dir():
            batch_annotations = parser_annotations / src_name
            batch_annotations.mkdir(parents=True, exist_ok=True)

            for json_path in batch_path.glob("*.json"):
                pdf_name = json_path.name.replace("json", "pdf")
                doc_path = GUIDELINES_DIR / src_name / pdf_name

                anno_file = _annotate_file(json_path, doc_path)
                anno_file.save(batch_annotations / pdf_name)
        else:
            raise ValueError(f"Error: Path {batch_path} does not exist or is not a directory.")

    # Single File logic
    else:
        pdf_name = f"{src_name}.pdf"
        json_path = parser_output / f"{src_name}.json"
        doc_path = GUIDELINES_DIR / pdf_name
        anno_path = parser_annotations / pdf_name
        anno_parent = anno_path.parent  # handle folder/file as the src_name

        anno_file = _annotate_file(json_path, doc_path)
        anno_parent.mkdir(parents=True, exist_ok=True)
        anno_file.save(anno_path)
