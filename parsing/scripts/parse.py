from typing import Any

from config import GUIDELINES_DIR
from parsing.methods.docling import DoclingParser
from parsing.methods.llamaparse import LlamaParseParser
from parsing.methods.config import Parsers
from parsing.methods.mineru import MinerUParser
from parsing.methods.unstructured import UnstructuredParser
from parsing.model.document_parser import DocumentParser
from parsing.model.options import ParserOptions
from parsing.scripts.get_parser import get_document_parser


def parse_pdf(
    parser_name: str,
    src_name: str,
    is_batch: bool = False,
    should_draw: bool = False,
    skip_existing=False,
):
    options = {
        ParserOptions.ANNOTATE: should_draw,
        ParserOptions.EXIST_OK: skip_existing,
    }

    parser_type = Parsers.get_parser_type(parser_name)
    parser = get_document_parser(parser_type)

    if is_batch:
        parser.process_batch(src_name, options)
    else:
        doc_path = GUIDELINES_DIR / f"{src_name}.pdf"
        parser.process_document(doc_path, options=options)
