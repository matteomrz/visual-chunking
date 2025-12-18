import logging
from pathlib import Path

from pymupdf import TEXTFLAGS_DICT, TEXT_PRESERVE_IMAGES, pymupdf, Document
from pymupdf.utils import (
    get_text,
    # get_textpage_ocr
)
from tqdm import tqdm

from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingResult, ParsingResultType

logger = logging.getLogger(__name__)

_pymupdf_flag = TEXTFLAGS_DICT & ~TEXT_PRESERVE_IMAGES

# Types for which we don't need spans
skip_types = [
    # Spans for TABLE_CELL already sufficient / more accurate
    ParsingResultType.TABLE,
    ParsingResultType.TABLE_ROW,
    # For now, no text from figures specifically needed
    ParsingResultType.FIGURE,
    # We don't want to split up title bounding boxes in our chunks
    ParsingResultType.TITLE,
    ParsingResultType.SECTION_HEADER,
]


def add_span_boxes(file_path: Path, root: ParsingResult):
    """
    Adds span-level bounding boxes to the ParsingResult using PyMuPDF.
    Traverses the root's children to add span-level bounding boxes if they are not present yet.

    Args:
        file_path: The path to the source PDF document.
        root: The root node of the parsed document structure.

    Raises:
        FileNotFoundError: If the ``file_path`` does not exist or is not a PDF file.
    """
    if not file_path.exists() or not file_path.name.endswith(".pdf"):
        raise FileNotFoundError(f"Error: Guideline PDF file not found at: {file_path}")

    doc = pymupdf.open(file_path)
    logger.info(f"Adding span-level bounding boxes for: {file_path.stem}")

    for child in tqdm(root.children):
        _add_spans_to_element(child, doc)


# TODO: Add filtering for element types (Don't do it for figures...)
def _add_spans_to_element(element: ParsingResult, pdf: Document):
    """
    Recursively extracts text spans using PyMuPDF from a PDF page cropped to the elements bounding boxes.
    The detected spans appended as children to their respective bounding box.

    Args:
        element: The current node in the parsing tree being processed.
        pdf: The open PyMuPDF document object used for extraction.
    """
    for child in element.children:
        _add_spans_to_element(child, pdf)

    if element.type in skip_types:
        return

    for bbox in element.geom:
        if not bbox.spans:
            page = pdf.load_page(bbox.page - 1)

            original_crop = page.cropbox
            height = page.cropbox.height
            width = page.cropbox.width

            # Fractional to absolute bounding boxes
            abs_left = bbox.left * width
            abs_top = bbox.top * height
            abs_right = bbox.right * width
            abs_bottom = bbox.bottom * height

            box_width = abs_right - abs_left
            box_height = abs_bottom - abs_top

            is_narrow = box_width < width * 0.02
            is_short = box_height < height * 0.01
            if is_short:
                # Short boxes do not need spans added to them
                continue
            elif is_narrow:  # TODO: Find out when too small boxes cause a problem
                logger.debug(
                    "Element Bounding Box is too small. Skipping Span creation. "
                    f"w={round(box_width, 3)}, h={round(box_height, 3)}."
                )
                continue

            # Crop the PDF page to the dimensions of the element bounding box
            rect = pymupdf.Rect(abs_left, abs_top, abs_right, abs_bottom)
            page.set_cropbox(rect)

            # Experimentally removed OCR textpage - Guidelines are all programmatic
            # Using lower dpi than 300 leads to scaling issues
            # tp = get_textpage_ocr(page, full=True, dpi=300)

            text = get_text(
                page,
                # textpage=tp,
                option="dict",
                sort=True,
                flags=_pymupdf_flag,
            )

            try:
                for block in text.get("blocks", []):
                    raw_lines = block.get("lines", [])

                    # Sometimes spans which only have very small intersect with element get retrieved
                    line_cnt = len(raw_lines)
                    for idx in reversed(range(line_cnt)):
                        line = raw_lines[idx]["bbox"]

                        # Remove line if any side extends more than threshold pixels out of the bounding box
                        threshold_pixels = 10
                        if (
                            line[0] < -threshold_pixels or
                            line[1] < -threshold_pixels or
                            line[2] > box_width + threshold_pixels or
                            line[3] > box_height + threshold_pixels
                        ):
                            raw_lines.pop(idx)

                    lines = _merge_adjacent_spans(raw_lines)

                    for line in lines:
                        # Raw line bounding boxes are relative to the element bounding boxes
                        # Transform to be relative to document dimensions
                        l_left = (rect.top_left.x + line[0]) / width
                        l_top = (rect.top_left.y + line[1]) / height
                        l_right = (rect.top_left.x + line[2]) / width
                        l_bottom = (rect.top_left.y + line[3]) / height

                        child_box = ParsingBoundingBox(
                            page=bbox.page,
                            left=l_left,
                            top=l_top,
                            right=l_right,
                            bottom=l_bottom,
                        )

                        bbox.spans.append(child_box)

            except (KeyError, ValueError) as e:
                logger.error(f"Unexpected data format in PyMuPDF output: {e}")
            except BaseException as e:
                logger.error(f"Unknown exception: {str(e)}")

            # Reset crop for next element
            page.set_cropbox(original_crop)


def _merge_adjacent_spans(
    lines: list[dict], overlap_limit: float = 0.5
) -> list[list[float]]:
    """
    Consolidates span bounding boxes based on significant vertical intersection.
    Two lines are merged if the overlap exceeds ``overlap_limit`` * the previous line's height.

    Args:
        lines: A list of dictionaries representing lines from PyMuPDF
        overlap_limit: Fractional value indicating the limit for a vertical intersect at which two lines are merged (Default: 0.5)

    Returns:
        A list of merged bounding box coordinates [l, t, r, b]
    """
    first_line = list(lines[0]["bbox"])
    merged_lines = [first_line]

    for raw_line in lines:
        line = raw_line["bbox"]

        last_line = merged_lines[-1]

        # Lines are ordered by the y-coordinates of the lower-right corner (from pymupdf.get_text())
        # Even if there is an intersect, the last_line must always lie higher than the new one
        intersect = last_line[3] - line[1]
        last_line_height = last_line[3] - last_line[1]

        if intersect > overlap_limit * last_line_height:
            last_line[0] = min(line[0], last_line[0])  # Leftest Left
            last_line[1] = min(line[1], last_line[1])  # Highest Top
            last_line[2] = max(line[2], last_line[2])  # Rightest Right
            last_line[3] = max(line[3], last_line[3])  # Lowest Bottom
        else:
            merged_lines.append(list(line))

    return merged_lines
