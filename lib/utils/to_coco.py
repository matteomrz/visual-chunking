from itertools import count
from pathlib import Path

import pymupdf

from lib.parsing.model.parsing_result import (
    ParsingResult,
    ParsingMetaData as PmD,
    ParsingResultType
)

# Create a simple id generator for coco annotation objects
id_counter = count(start=1)

# We only need bounding boxes of high-level elements and can ignore more granular bounding boxes
skip_children_types = [
    ParsingResultType.TABLE,
    ParsingResultType.DOC_INDEX,
    ParsingResultType.LIST,
    ParsingResultType.REFERENCE_LIST,
    ParsingResultType.KEY_VALUE_AREA
]


def get_coco_annotations(root: ParsingResult) -> list[dict]:
    """
    Transforms the Parsing Result into a list of COCO annotations.
    Only applicable for single page documents as from PubLayNet.

    Args:
        root: Root element of the ParsingResult

    Returns:
        List of COCO annotation dictionaries. Format: https://cocodataset.org/#format-data
    """

    pdf_path = Path(root.metadata[PmD.GUIDELINE_PATH.value])
    img_id = int(pdf_path.stem)

    pdf_doc = pymupdf.open(pdf_path).load_page(0).cropbox
    height = pdf_doc.height
    width = pdf_doc.width

    annotations = [e for e in _get_coco(root, height, width, img_id)]
    return _filter_text_in_figures(annotations)


def _get_coco(elem: ParsingResult, p_height: int, p_width: int, img_id: int):
    """
    Recursively generates COCO annotations in reading order.
    If an elements geometry is made up of multiple bounding boxes, multiple annotations are returned.

    Args:
        elem: The ParsingResult to transform
        p_height: Height of the page
        p_width: Width of the page
        img_id: Document Identifier
    """

    if elem.type != ParsingResultType.ROOT:
        for i, parsing_box in enumerate(elem.geom):
            anno_id = next(id_counter)
            category_id = _get_category_id(elem)

            x = parsing_box.left * p_width
            y = parsing_box.top * p_height
            width = parsing_box.right * p_width - x
            height = parsing_box.bottom * p_height - y

            area = width * height
            bbox = [x, y, width, height]

            # Temporary flag to be able to filter out unneeded text from figures
            is_caption = elem.type == ParsingResultType.CAPTION

            annotation = {
                "id": anno_id,
                "image_id": img_id,
                "category_id": category_id,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": 1,  # We do not have scores for the prediction TODO
                "is_caption": is_caption
            }

            yield annotation

    if elem.type not in skip_children_types:
        for child in elem.children:
            yield from _get_coco(child, p_height, p_width, img_id)


def _get_category_id(elem: ParsingResult) -> int:
    """Maps the ParsingResultType to the PubLayNet category_id."""

    match elem.type:
        # title
        case ParsingResultType.TITLE | ParsingResultType.SECTION_HEADER:
            return 2
        # list
        case ParsingResultType.LIST | ParsingResultType.LIST_ITEM | ParsingResultType.REFERENCE_LIST | \
             ParsingResultType.REFERENCE_ITEM | ParsingResultType.KEY_VALUE_AREA:
            return 3
        # table
        case ParsingResultType.TABLE | ParsingResultType.TABLE_ROW | ParsingResultType.TABLE_CELL | \
             ParsingResultType.DOC_INDEX:
            return 4
        # figure
        case ParsingResultType.FIGURE:
            return 5
        # Return text on default
        case _:
            return 1


def _filter_text_in_figures(annotations: list) -> list:
    """
    Often texts inside of figures are also recognized.
    As these are not expected in the ground truth, we remove them.
    """

    figure_boxes = [c["bbox"] for c in annotations if c["category_id"] == 5]

    filtered = []
    for coco in annotations:
        if coco["category_id"] == 1 and not coco["is_caption"]:
            bbox = coco["bbox"]
            if any(
                bbox[0] >= f[0] and bbox[1] >= f[1] and bbox[0] + bbox[2] <= f[0] + f[2] and bbox[
                    1] + bbox[3] <= f[1] + f[3] for f in figure_boxes):
                continue

        del coco["is_caption"]
        filtered.append(coco)

    return filtered
