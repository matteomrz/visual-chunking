import logging
from pathlib import Path
from typing import Any

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    DocItem,
    FloatingItem,
    GroupItem,
    NodeItem,
    PageItem,
    SectionHeaderItem,
    TableCell,
    TableItem,
    TextItem,
)

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingResult,
    ParsingResultType,
    ParsingMetaData as PmD,
)

logger = logging.getLogger(__name__)


class DoclingParser(DocumentParser[DoclingDocument]):
    """Uses the Docling Package for parsing PDF documents"""

    label_mapping = {
        # Texts
        "TEXT": ParsingResultType.PARAGRAPH,
        "SECTION_HEADER": ParsingResultType.SECTION_HEADER,
        "FOOTNOTE": ParsingResultType.FOOTNOTE,
        # Lists
        "LIST": ParsingResultType.LIST,
        "LIST_ITEM": ParsingResultType.LIST_ITEM,
        # Figures and Tables
        "PICTURE": ParsingResultType.FIGURE,
        "CAPTION": ParsingResultType.CAPTION,
        "TABLE": ParsingResultType.TABLE,
        "DOCUMENT_INDEX": ParsingResultType.DOC_INDEX,
        "TABLE_ROW": ParsingResultType.TABLE_ROW,
        "TABLE_CELL": ParsingResultType.TABLE_CELL,
        # Miscellaneous
        "PAGE_HEADER": ParsingResultType.PAGE_HEADER,
        "PAGE_FOOTER": ParsingResultType.PAGE_FOOTER,
        "KEY_VALUE_AREA": ParsingResultType.KEY_VALUE_AREA,
        "FORM_AREA": ParsingResultType.FORM_AREA,
    }

    def __init__(self, use_vlm=False):
        if use_vlm:
            self.module = Parsers.DOCLING_GRANITE

            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX
            )
            pdf_format_option = PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            )

        else:
            self.module = Parsers.DOCLING

            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                force_backend_text=False,
                do_ocr=True,
                table_structure_options=TableStructureOptions(
                    do_cell_matching=True, mode=TableFormerMode.ACCURATE
                ),
            )
            pdf_format_option = PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, pipeline_options=pipeline_options
            )

        self.parser = DocumentConverter(
            format_options={InputFormat.PDF: pdf_format_option}
        )

    def _parse(self, file_path: Path, options: dict = None) -> DoclingDocument:
        result = self.parser.convert(file_path)
        return result.document

    def _transform(self, raw_result: DoclingDocument) -> ParsingResult:
        root = ParsingResult.root()
        docling_root = raw_result.body
        for child in docling_root.children:
            self._transform_item(root, raw_result, child.resolve(raw_result))

        return root

    def _get_md(self, raw_result: DoclingDocument, file_path: Path) -> str:
        return raw_result.export_to_markdown()

    def _transform_item(self, res: ParsingResult, doc: DoclingDocument, item: Any):
        if not isinstance(item, NodeItem):
            logger.error(f"Item is not a valid NodeItem. Actual: {type(item)}")
            return

        item_id = item.self_ref

        # Additional metadata saved with the element
        meta = {}

        # Handle GroupItem
        if isinstance(item, GroupItem):
            item_type = self._get_element_type(item.label.name)

            transformed = ParsingResult(
                id=item_id, content='', type=item_type, geom=[], parent=res
            )

        # Handle DocItem
        elif isinstance(item, DocItem):
            item_type = self._get_element_type(item.label.name)

            # Handle TextItem
            if isinstance(item, TextItem):
                if isinstance(item, SectionHeaderItem):
                    level = item.level
                    # Currently Docling only gives out Headers of Level 1
                    # If this changes, we can use the header level to correct the
                    # hierarchy during post-processing
                    meta[PmD.HEADER_LEVEL.value] = level
                item_content = item.text

            # Handle FloatingItem
            elif isinstance(item, FloatingItem):
                item_content = item.caption_text(doc)

            # Unhandled DocItem Type
            else:
                logger.warning(f"Unhandled DocItem type: {type(item)}")
                item_content = ""

            transformed = ParsingResult(
                id=item_id,
                content=item_content,
                type=item_type,
                parent=res,
                metadata=meta,
                geom=[
                    _transform_b_box(prov.bbox, doc.pages[prov.page_no])
                    for prov in item.prov
                ],
            )

        # Unhandled NodeItem Type
        else:
            logger.warning(f"Unhandled NodeItem type: {type(item)}")
            return

        if isinstance(item, TableItem):
            page = doc.pages[transformed.geom[0].page]
            _transform_table(transformed, page, item.data.grid)

        for child in item.children:
            self._transform_item(transformed, doc, child.resolve(doc))
        res.children.append(transformed)


def _transform_table(
    table: ParsingResult, page: PageItem, grid: list[list[TableCell]]
):
    parent_id = table.id

    for idx, row in enumerate(grid):
        cell_results = []
        min_l = None
        min_t = None
        max_r = None
        max_b = None

        for cell in row:
            if not cell.bbox:
                continue

            x = cell.start_col_offset_idx
            y = cell.start_row_offset_idx

            # If there is a merged cell it will appear multiple times with the same index
            if x < len(cell_results):
                continue

            box = _transform_b_box(cell.bbox, page)

            if min_l is None or box.left < min_l:
                min_l = box.left
            if min_t is None or box.top < min_t:
                min_t = box.top
            if max_r is None or box.right > max_r:
                max_r = box.right
            if max_b is None or box.bottom > max_b:
                max_b = box.bottom

            cell_res = ParsingResult(
                id=f"{parent_id}_{y}_{x}",
                type=ParsingResultType.TABLE_CELL,
                content=cell.text,
                geom=[box],
                parent=None,
            )

            cell_results.append(cell_res)

        row_box = ParsingBoundingBox(
            left=min_l, top=min_t, right=max_r, bottom=max_b, page=page.page_no
        )

        row_res = ParsingResult(
            id=f"{parent_id}_{idx}",
            type=ParsingResultType.TABLE_ROW,
            content=" | ".join([c.content for c in cell_results]),
            parent=table,
            children=cell_results,
            geom=[row_box],
        )

        for cell in cell_results:
            cell.parent = row_res
        table.children.append(row_res)


def _transform_b_box(raw_b_box: BoundingBox, page: PageItem) -> ParsingBoundingBox:
    page_height = page.size.height
    page_width = page.size.width
    raw_b_box = raw_b_box.to_top_left_origin(page_height=page_height)

    l_frac = max(0.0, raw_b_box.l / page_width)
    t_frac = max(0.0, raw_b_box.t / page_height)
    r_frac = max(0.0, raw_b_box.r / page_width)
    b_frac = max(0.0, raw_b_box.b / page_height)

    return ParsingBoundingBox(
        page=page.page_no, left=l_frac, top=t_frac, right=r_frac, bottom=b_frac
    )
