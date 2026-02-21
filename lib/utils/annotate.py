import logging
from pathlib import Path

from pymupdf import Document, Page, pymupdf

from config import ANNOTATED_DIR, CHUNKING_RESULT_DIR, PARSING_RESULT_DIR
from lib.chunking.model.chunk import ChunkingResult
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingMetaData as PmD, \
    ParsingResult

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


def _draw_box(
    box: ParsingBoundingBox,
    page: Page,
    color: tuple,
    label: str = "",
    fill: bool = False,
    border: bool = True
):
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
        if not border:
            color = None
    else:
        fill_color = None

    page.draw_rect(
        rect=rect,
        color=color,
        fill=fill_color,
        width=1.5,
        fill_opacity=0.3,
        stroke_opacity=1.0
    )

    if label:
        page.insert_text(
            point=(l, t - 3),
            render_mode=2,
            text=label,
            fontsize=6,
            color=color,
            border_width=0.02,
            fill=(0, 0, 0),
        )


def _draw_parsing_result(result: ParsingResult, doc: Document):
    loaded_idx = -1
    page = None

    for box in result.geom:
        page_idx = box.page - 1

        if page_idx < 0 or page_idx >= doc.page_count:
            logger.warning(f"Malformed page number `{page_idx + 1}` in {str(box)}")
            continue

        if loaded_idx != page_idx:
            page = doc.load_page(page_idx)

        label = result.type.value
        color = _get_color(label)

        _draw_box(box, page, color, label=label)

        # Draw spans if available
        for child in box.spans:
            _draw_box(child, page, color, fill=True, border=False)

    for child in result.children:
        _draw_parsing_result(child, doc)


def _draw_chunking_result(result: ChunkingResult, doc: Document):
    loaded_idx = -1
    page = None

    chunk_colors = [
        (0.1216, 0.4667, 0.7059),
        (1.0000, 0.4980, 0.0549)
    ]

    for idx, chunk in enumerate(result.chunks):
        for box in chunk.geom:
            page_idx = box.page - 1

            if page_idx < 0 or page_idx >= doc.page_count:
                logger.warning(f"Malformed page number `{page_idx + 1}` in {str(box)}")
                continue

            if loaded_idx != page_idx:
                page = doc.load_page(page_idx)

            color = chunk_colors[idx % 2]
            _draw_box(box, page, color, label=chunk.id, fill=True)


def create_annotation(annotation_object: ParsingResult | ChunkingResult):
    try:
        pdf_path = Path(annotation_object.metadata[PmD.GUIDELINE_PATH.value])
    except KeyError:
        raise ValueError(f"The annotated element is missing the PDF path set in its metadata.")

    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)

    if isinstance(annotation_object, ParsingResult):
        _draw_parsing_result(annotation_object, doc)
        json_path = Path(annotation_object.metadata[PmD.JSON_PATH.value])
        output_path = json_path.relative_to(PARSING_RESULT_DIR)
    else:
        _draw_chunking_result(annotation_object, doc)
        json_path = Path(annotation_object.metadata["chunk_path"])
        output_path = json_path.relative_to(CHUNKING_RESULT_DIR)

    output_path = (ANNOTATED_DIR / output_path).with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc.save(output_path)

    logger.info(f"Saved annotated PDF document to: {output_path}")
