from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    DocItem,
    FloatingItem,
    GroupItem,
    NodeItem,
    PageItem,
    TextItem,
)

from parsing.methods.config import Parsers
from parsing.model.document_parser import DocumentParser
from parsing.model.parsing_result import ParsingBoundingBox, ParsingResult


class DoclingParser(DocumentParser[DoclingDocument]):
    """Uses the Docling Package for parsing PDF documents"""

    module = Parsers.DOCLING

    def __init__(self):
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
            _transform_item(root, raw_result, child.resolve(raw_result))

        return root


def _transform_item(res: ParsingResult, doc: DoclingDocument, item: Any):
    if not isinstance(item, NodeItem):
        print(f"Error: Item is not a valid NodeItem. Actual: {type(item)}")
        return

    # Handle GroupItem
    if isinstance(item, GroupItem):
        transformed = ParsingResult(
            id=item.self_ref, content=item.name, type=item.label.name, geom=[]
        )

    # Handle DocItem
    elif isinstance(item, DocItem):
        # Handle TextItem
        if isinstance(item, TextItem):
            item_content = item.text

        # Handle FloatingItem
        elif isinstance(item, FloatingItem):
            item_content = item.caption_text(doc)

        # Unhandled DocItem Type
        else:
            print(f"Unhandled DocItem type: {type(item)}")
            item_content = ""

        transformed = ParsingResult(
            id=item.self_ref,
            content=item_content,
            type=item.label.name,
            geom=[
                _transform_b_box(prov.bbox, doc.pages[prov.page_no])
                for prov in item.prov
            ],
        )

    # Unhandled NodeItem Type
    else:
        print(f"Unhandled NodeItem type: {type(item)}")
        return

    for child in item.children:
        _transform_item(transformed, doc, child.resolve(doc))
    res.children.append(transformed)


def _transform_b_box(raw_b_box: BoundingBox, page: PageItem) -> ParsingBoundingBox:
    page_height = page.size.height
    page_width = page.size.width
    raw_b_box = raw_b_box.to_top_left_origin(page_height=page_height)

    l_frac = raw_b_box.l / page_width
    t_frac = raw_b_box.t / page_height
    r_frac = raw_b_box.r / page_width
    b_frac = raw_b_box.b / page_height

    return ParsingBoundingBox(
        page=page.page_no,
        left=l_frac,
        top=t_frac,
        right=r_frac,
        bottom=b_frac
    )
