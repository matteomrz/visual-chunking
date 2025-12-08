from typing import Any

from config import GUIDELINES_DIR
from parsing.methods.docling import DoclingParser
from parsing.methods.llamaparse import LlamaParseParser
from parsing.methods.config import Parsers
from parsing.methods.mineru import MinerUParser
from parsing.methods.unstructured import UnstructuredParser
from parsing.model.document_parser import DocumentParser
from parsing.model.options import ParserOptions


def _get_parser(parser_name: str, options: dict) -> DocumentParser[Any]:
    match Parsers.get_parser(parser_name):
        case Parsers.LLAMA_PARSE:
            return LlamaParseParser()
        case Parsers.DOCLING:
            return DoclingParser()
        case Parsers.UNSTRUCTURED_IO:
            return UnstructuredParser()
        case Parsers.MINERU_PIPELINE:
            return MinerUParser(use_vlm=False)
        case Parsers.MINERU_VLM:
            return MinerUParser(use_vlm=True)
        case _:
            raise ValueError(f'No DocumentParser specified for type "{parser_name}"')


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

    parser = _get_parser(parser_name, options)

    if is_batch:
        parser.process_batch(src_name, options)
    else:
        doc_path = GUIDELINES_DIR / f"{src_name}.pdf"
        parser.process_document(doc_path, options=options)
