import json
import os
from pathlib import Path

from mineru.backend.vlm.vlm_analyze import ModelSingleton, doc_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter

from config import GUIDELINES_DIR
from parsing.methods.config import Parsers
from parsing.model.document_parser import DocumentParser
from parsing.model.parsing_result import ParsingBoundingBox, ParsingResult


class MinerUParser(DocumentParser):
    """Uses the MinerU Package for parsing PDF documents"""

    module = Parsers.MINERU

    def __init__(self):
        self.model = ModelSingleton().get_model(
            backend="mlx-engine", model_path=None, server_url=None)

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        # Prepare Image directory
        file_image_path = self.image_path / file_path.stem
        file_image_path.mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(parent_dir=str(file_image_path))

        pdf_bytes = _get_pdf_bytes(file_path)

        result, _ = doc_analyze(pdf_bytes, image_writer=image_writer, predictor=self.model)

        return result

    def _transform(self, raw_result: dict) -> ParsingResult:
        root = ParsingResult.root()
        pages = raw_result.get("pdf_info", [])
        for page in pages:
            if not isinstance(page, dict):
                print(f"Warning: Wrong page type. Expected `dict`, Actual `{type(page)}`")
                continue

            for element in page.get("para_blocks", []):
                _transform_element(parent=root, element=element, page=page)

        return root


def _transform_element(parent: ParsingResult, element: dict, page: dict):
    if not isinstance(element, dict):
        print(f"Warning: Wrong element type. Expected `dict`, Actual `{type(element)}`")
        return

    content = _get_content(element)
    elem_type = element.get("type", "undefined")
    elem_type = elem_type + "_" + element.get("sub_type", "")
    idx = element.get("index", "")

    b_box = _get_bounding_box(element, page)

    result = ParsingResult(
        id=f"{elem_type}_{idx}",
        type=elem_type,
        content=content,
        geom=[b_box]
    )

    parent.children.append(result)
    child_nodes = element.get("blocks", [])
    for node in child_nodes:
        _transform_element(parent=result, element=node, page=page)


def _get_pdf_bytes(file_path: Path):
    """Transform PDF into required input data format"""
    pdf_bytes = read_fn(file_path)
    return convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)


def _get_content(element: dict) -> str:
    """Collect element content from the constructing lines."""
    content = ""
    lines = element.get("lines", [])
    for line in lines:
        if not isinstance(line, dict):
            continue

        spans = line.get("spans", [])
        for span in spans:
            if not isinstance(span, dict):
                continue

            content = content + span.get("content", "")
    return content


def _get_bounding_box(element: dict, page: dict) -> ParsingBoundingBox:
    """Parse Bounding box to needed format."""

    page_nr = page.get("page_idx", 0) + 1

    page_size = page.get("page_size", [])
    if len(page_size) < 2:
        print(f"Warning: Malformed Page Dimensions: {page_size}")
        page_size = [1.0] * 2

    b_box = element.get("bbox", [])
    if len(b_box) < 4:
        print(f"Warning: Malformed Bounding Box: {b_box}")
        b_box = [0.0] * 4

    # Could also return a list of all line bounding boxes
    return ParsingBoundingBox(
        page=page_nr,
        left=b_box[0] / page_size[0],
        top=b_box[1] / page_size[1],
        right=b_box[2] / page_size[0],
        bottom=b_box[3] / page_size[1]
    )
