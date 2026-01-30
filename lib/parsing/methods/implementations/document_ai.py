import os
from pathlib import Path

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as document_ai

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingMetaData as PmD,
    ParsingResult,
    ParsingResultType
)

DocumentLayoutBlock = document_ai.Document.DocumentLayout.DocumentLayoutBlock

# Document AI Parsing Configuration
PDF_MIME_TYPE = "application/pdf"

layout_config = document_ai.types.ProcessOptions.LayoutConfig()
layout_config.return_bounding_boxes = True

PROCESS_OPTIONS = document_ai.ProcessOptions()
PROCESS_OPTIONS.layout_config = layout_config


# TODO: Give more info about how to use document ai
class DocumentAIParser(DocumentParser[document_ai.Document]):
    """Uses the LayoutParser from Google's Document AI for parsing."""

    module = Parsers.DOCUMENT_AI

    # Document AI only uses type strings for Text blocks
    # Also has types for ordered and unordered lists, however we do not include these for now
    label_mapping = {
        "paragraph": ParsingResultType.PARAGRAPH,
        "subtitle": ParsingResultType.CAPTION,
        "heading": ParsingResultType.SECTION_HEADER,
        "header": ParsingResultType.PAGE_HEADER,
        "footer": ParsingResultType.PAGE_FOOTER,
    }

    client: document_ai.DocumentProcessorServiceClient

    layout_path: str
    ocr_path: str

    def __init__(self):
        load_dotenv()
        location = os.getenv("DOC_AI_LOCATION")
        project = os.getenv("DOC_AI_PROJECT_ID")
        layout_processor = os.getenv("DOC_AI_LAYOUT_PROCESSOR_ID")
        ocr_processor = os.getenv("DOC_AI_OCR_PROCESSOR_ID")

        client_options = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )

        self.client = document_ai.DocumentProcessorServiceClient(
            client_options=client_options
        )

        self.layout_path = self.client.processor_path(project, location, layout_processor)
        self.ocr_path = self.client.processor_path(project, location, ocr_processor)

    def _request_doc_ai(self, file_path: Path, processor_path: str) -> document_ai.Document:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        raw_doc = document_ai.RawDocument()
        raw_doc.content = file_bytes
        raw_doc.mime_type = PDF_MIME_TYPE

        request = document_ai.ProcessRequest()
        request.name = processor_path
        request.raw_document = raw_doc
        request.process_options = PROCESS_OPTIONS

        response = self.client.process_document(request)

        return response.document

    def _parse(self, file_path: Path, options: dict = None):
        return self._request_doc_ai(file_path, self.layout_path)

    def _get_md(self, raw_result: document_ai.Document, file_path: Path) -> str:
        document = self._request_doc_ai(file_path, self.ocr_path)
        return document.text

    def _transform(self, raw_result: document_ai.Document) -> ParsingResult:
        root = ParsingResult.root()

        blocks = raw_result.document_layout.blocks
        for block in blocks:
            block_res = self._transform_block(block)
            block_res.parent = root
            root.children.append(block_res)

        return root

    def _transform_block(self, block: DocumentLayoutBlock) -> ParsingResult:
        elem_id = block.block_id

        metadata = {}
        children: list[ParsingResult] = []

        # Only one of the block fields is set at the same time
        # List transformation
        if block.list_block:
            item = block.list_block

            elem_type = ParsingResultType.LIST
            elem_content = ""

            for entry in item.list_entries:
                entry_res = [
                    self._transform_block(entry_block)
                    for entry_block in entry.blocks
                ]

                children.extend(entry_res)

        # Table transformation
        elif block.table_block:
            item = block.table_block

            elem_type = ParsingResultType.TABLE
            elem_content = item.caption

            children.extend(self._transform_table(block))

        # Text transformation
        else:
            item = block.text_block
            elem_content = item.text

            raw_type = item.type_
            splits = raw_type.split("-")

            elem_type = self._get_element_type(splits[0])

            if elem_type == ParsingResultType.SECTION_HEADER and len(splits) > 1:
                level_str = splits[-1]
                heading_level = int(level_str) if level_str.isdigit() else 0
                metadata[PmD.HEADER_LEVEL.value] = heading_level

            for child in item.blocks:
                block_res = self._transform_block(child)
                children.append(block_res)

        elem_geom = _get_bounding_box(block)

        elem_res = ParsingResult(
            id=elem_id,
            type=elem_type,
            children=children,
            content=elem_content,
            metadata=metadata,
            parent=None,
            geom=elem_geom
        )

        for child in children:
            child.parent = elem_res

        return elem_res

    def _transform_table(self, block: DocumentLayoutBlock) -> list[ParsingResult]:
        item = block.table_block
        elem_id = block.block_id

        rows = []
        rows.extend(item.header_rows)
        rows.extend(item.body_rows)

        rows_res = []

        for row_idx, row in enumerate(rows):

            min_l = None
            min_t = None
            max_r = None
            max_b = None

            cells_res: list[ParsingResult] = []

            for cell_idx, cell in enumerate(row.cells):

                blocks_res = []
                for block in cell.blocks:
                    block_res = self._transform_block(block)
                    blocks_res.append(block_res)

                    for box in block_res.geom:
                        if min_l is None or box.left < min_l:
                            min_l = box.left
                        if min_t is None or box.top < min_t:
                            min_t = box.top
                        if max_r is None or box.right > max_r:
                            max_r = box.right
                        if max_b is None or box.bottom > max_b:
                            max_b = box.bottom

                cell_id = f"{elem_id}_{row_idx}_{cell_idx}"
                cell_geom = [b for res in blocks_res for b in res.geom]

                cell_res = ParsingResult(
                    content="",
                    type=ParsingResultType.TABLE_CELL,
                    id=cell_id,
                    geom=cell_geom,
                    children=blocks_res,
                    parent=None,
                )
                cells_res.append(cell_res)

            row_id = f"{elem_id}_{row_idx}"
            row_box = ParsingBoundingBox(
                page=block.page_span.page_start,
                left=min_l,
                top=min_t,
                right=max_r,
                bottom=max_b
            )

            row_res = ParsingResult(
                content="",
                type=ParsingResultType.TABLE_ROW,
                id=row_id,
                geom=[row_box],
                parent=None
            )

            for cell_res in cells_res:
                cell_res.parent = row_res

            rows_res.append(row_res)

        return rows_res


def _get_bounding_box(block: DocumentLayoutBlock) -> list[ParsingBoundingBox]:
    raw_box = block.bounding_box.normalized_vertices
    coord_cnt = len(raw_box)

    # TODO: How can i know when we switch pages? Check if bounding box is higher than last?
    start_page = block.page_span.page_start

    geom = []

    i = 0
    while i * 4 < coord_cnt:
        start_idx = i * 4
        points = raw_box[start_idx: start_idx + 4]

        top_left = points[0]
        bottom_right = points[2]

        bbox = ParsingBoundingBox(
            page=start_page,
            left=top_left.x,
            top=top_left.y,
            right=bottom_right.x,
            bottom=bottom_right.y
        )

        geom.append(bbox)
        i += 1

    return geom
