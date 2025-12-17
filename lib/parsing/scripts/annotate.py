import logging
from pathlib import Path

from pymupdf import Document, Page, pymupdf

from config import ANNOTATED_DIR, BOUNDING_BOX_DIR, GUIDELINES_DIR
from lib.parsing.methods.config import Parsers
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingResult
from lib.utils.open import open_parsing_result

logger = logging.getLogger(__name__)

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


def _draw_box(box: ParsingBoundingBox, page: Page, color: tuple, opacity: float = 1.0,
              fill: bool = False):
    page_size = page.rect
    page_height = page_size.height
    page_width = page_size.width

    l = box.left * page_width
    t = box.top * page_height
    r = box.right * page_width
    b = box.bottom * page_height

    rect = pymupdf.Rect(l, t, r, b)

    if fill:
        fill_color = color
        color = None
    else:
        fill_color = None

    page.draw_rect(
        rect=rect,
        color=color,
        fill=fill_color,
        width=1.5,
        fill_opacity=opacity,
        stroke_opacity=opacity
    )


def _draw_element(element: ParsingResult, doc: Document):
    loaded_idx = -1
    page = None

    for box in element.geom:
        page_idx = box.page - 1

        if page_idx < 0 or page_idx >= doc.page_count:
            logger.warning(f"Malformed page number `{page_idx + 1}` in {str(box)}")
            continue

        if loaded_idx != page_idx:
            page = doc.load_page(page_idx)

        page_size = page.rect
        page_height = page_size.height
        page_width = page_size.width

        l = box.left * page_width
        t = box.top * page_height

        label = element.type.value
        color = _get_color(label)

        _draw_box(box, page, color)
        page.insert_text(
            point=(l, t - 3),
            text=label,
            fontsize=6,
            fill=color,
            fill_opacity=0.6
        )

        # Draw spans if available
        for child in box.spans:
            _draw_box(child, page, color, opacity=0.3, fill=True)

    for child in element.children:
        _draw_element(child, doc)


def _annotate_file(json_path: Path, doc_path: Path) -> Document:
    if not doc_path.exists():
        raise ValueError(f"File not found: {doc_path}")

    document = pymupdf.open(doc_path)
    result = open_parsing_result(json_path)

    for elem in result.children:
        _draw_element(elem, document)

    return document


def create_annotation(src_path: Path, parser: Parsers | str):
    if isinstance(parser, str):
        parser = Parsers.get_parser_type(parser)

    if not src_path.exists():
        raise ValueError(f"The source path does not exist: {src_path}")

    is_batch = src_path.is_dir()

    if is_batch:
        dir_path = src_path
    elif src_path.name.endswith(".pdf"):
        dir_path = src_path.parent
    else:
        raise ValueError(f"No PDF file found at {src_path}")

    rel_path = dir_path.relative_to(GUIDELINES_DIR)
    anno_dir = ANNOTATED_DIR / parser.value / rel_path
    bbox_dir = BOUNDING_BOX_DIR / parser.value / rel_path

    if not bbox_dir.exists() or not bbox_dir.is_dir():
        raise ValueError(f"Error: Directory {bbox_dir} does not exist or is not a directory.")

    anno_dir.mkdir(parents=True, exist_ok=True)

    # Batch Logic
    if is_batch:
        for json_path in bbox_dir.glob("*.json"):
            pdf_name = json_path.stem + ".pdf"
            doc_path = src_path / pdf_name

            anno_file = _annotate_file(json_path, doc_path)
            anno_path = anno_dir / pdf_name
            anno_file.save(anno_dir / pdf_name)
            logger.info(f"Saved annotated PDF document to: {anno_path}")


    # Single File logic
    else:
        json_path = bbox_dir / f"{src_path.stem}.json"

        anno_file = _annotate_file(json_path, src_path)
        anno_path = anno_dir / src_path.name
        anno_file.save(anno_path)
        logger.info(f"Saved annotated PDF document to: {anno_path}")
