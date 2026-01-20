import logging
from pathlib import Path
from typing import Any

from mineru.backend.vlm.vlm_analyze import (
    ModelSingleton as VlmModelSingleton,
    doc_analyze as vlm_doc_analyze,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.pipeline.pipeline_analyze import (
    ModelSingleton as PipelineModelSingleton,
    doc_analyze as pipeline_doc_analyze,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_get_middle_json,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingResult,
    ParsingResultType,
)

logger = logging.getLogger(__name__)


class MinerUParser(DocumentParser):
    """Uses the MinerU Package for parsing PDF documents"""

    label_mapping = {
        # Texts
        "text": ParsingResultType.PARAGRAPH,
        "title": ParsingResultType.SECTION_HEADER,
        "interline_equation": ParsingResultType.FORMULA,
        # Lists
        "list": ParsingResultType.LIST,
        "list_text": ParsingResultType.LIST,
        "list_ref_text": ParsingResultType.REFERENCE_LIST,
        "ref_text": ParsingResultType.REFERENCE_ITEM,
        # Figures and Tables
        "image": ParsingResultType.FIGURE,
        "image_body": ParsingResultType.FIGURE,
        "image_caption": ParsingResultType.CAPTION,
        "table_caption": ParsingResultType.CAPTION,
        "table": ParsingResultType.TABLE,
        "table_body": ParsingResultType.TABLE,
        "table_footnote": ParsingResultType.FOOTNOTE,
    }

    def __init__(self, use_vlm=False):
        self.is_vlm = use_vlm

        if self.is_vlm:
            self.module = Parsers.MINERU_VLM
            vlm_type = get_vlm_engine("auto")  # Automatically determine correct vlm engine
            self.model = VlmModelSingleton().get_model(
                backend=vlm_type, model_path=None, server_url=None
            )
        else:
            self.module = Parsers.MINERU_PIPELINE
            self.model = PipelineModelSingleton().get_model()

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        file_image_path = self._create_image_dir(file_path)
        image_writer = FileBasedDataWriter(parent_dir=str(file_image_path))

        pdf_bytes = _get_pdf_bytes(file_path)
        if self.is_vlm:
            result, _ = vlm_doc_analyze(
                pdf_bytes, image_writer=image_writer, predictor=self.model
            )
        else:
            result = _get_pipeline_result(pdf_bytes, image_writer)

        return result

    def _transform(self, raw_result: dict) -> ParsingResult:
        root = ParsingResult.root()
        pages = raw_result.get("pdf_info", [])
        for page in pages:
            if not isinstance(page, dict):
                logging.warning(f"Wrong page type. Expected `dict`, Actual `{type(page)}`")
                continue

            for element in page.get("para_blocks", []):
                self._transform_element(parent=root, element=element, page=page)

        return root

    def _get_md(self, raw_result: dict, file_path: Path) -> str:
        pdf_info = raw_result.get("pdf_info", [])
        image_dir = self._create_image_dir(file_path)

        if self.is_vlm:
            return vlm_union_make(
                pdf_info, MakeMode.MM_MD, img_buket_path=str(image_dir)
            )
        else:
            return pipeline_union_make(
                pdf_info, MakeMode.MM_MD, img_buket_path=str(image_dir)
            )

    def _transform_element(self, parent: ParsingResult, element: dict, page: dict):
        if not isinstance(element, dict):
            logging.warning(f"Wrong element type. Expected `dict`, Actual `{type(element)}`")
            return

        content = _get_content(element)

        elem_type = element.get("type", "unknown")
        if "sub_type" in element.keys():
            elem_type += f"_{element['sub_type']}"

        parsed_type = self._get_element_type(elem_type)
        idx = element.get("index", "")

        # Line output only works on MinerU Pipeline
        with_spans = self.module == Parsers.MINERU_PIPELINE
        b_box = _get_bounding_box(element, page, with_spans)

        result = ParsingResult(
            id=f"{parsed_type.value}_{idx}",
            type=parsed_type,
            content=content,
            geom=[b_box],
            parent=parent,
        )

        parent.children.append(result)
        child_nodes = element.get("blocks", [])
        for node in child_nodes:
            self._transform_element(parent=result, element=node, page=page)


def _get_pdf_bytes(file_path: Path) -> bytes | Any:
    """Transform PDF into required input data format"""
    pdf_bytes = read_fn(file_path)
    return convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)


def _get_pipeline_result(
    pdf_bytes: bytes | Any, image_writer: FileBasedDataWriter
) -> dict:
    infer_result, all_images, all_pdfs, lang_list, ocr_list = pipeline_doc_analyze(
        [pdf_bytes], ["en"]
    )

    logger.debug(f"Results count from pipeline: {len(infer_result)}")

    return pipeline_get_middle_json(
        infer_result[0],
        all_images[0],
        all_pdfs[0],
        image_writer,
        lang_list[0],
        ocr_list[0],
    )


def _get_content(element: dict) -> str:
    """Collect element content from the constructing spans."""
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


def _get_bounding_box(
    element: dict, page: dict, with_lines: bool
) -> ParsingBoundingBox:
    """Parse Bounding box to needed format."""
    page_nr = page.get("page_idx", 0) + 1

    page_size = page.get("page_size", [])
    if len(page_size) < 2:
        logger.warning(f"Malformed Page Dimensions: {page_size}")
        page_size = [1.0] * 2

    b_box = element.get("bbox", [])
    if len(b_box) < 4:
        logger.warning(f"Malformed Bounding Box: {b_box}")
        b_box = [0.0] * 4

    # Transform Line Bounding Boxes
    line_boxes = []
    if with_lines:
        lines = element.get("lines", [])
        for line in lines:
            l_box = _get_bounding_box(line, page, with_lines)
            line_boxes.append(l_box)

    return ParsingBoundingBox(
        page=page_nr,
        left=b_box[0] / page_size[0],
        top=b_box[1] / page_size[1],
        right=b_box[2] / page_size[0],
        bottom=b_box[3] / page_size[1],
        spans=line_boxes,
    )
