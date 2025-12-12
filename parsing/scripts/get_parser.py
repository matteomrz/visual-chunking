from typing import Any

from parsing.methods.config import Parsers
from parsing.methods.docling import DoclingParser
from parsing.methods.llamaparse import LlamaParseParser
from parsing.methods.mineru import MinerUParser
from parsing.methods.unstructured import UnstructuredParser
from parsing.model.document_parser import DocumentParser


def get_document_parser(parser_type: Parsers) -> DocumentParser[Any]:
    match parser_type:
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
            raise ValueError(f'No DocumentParser specified for type "{parser_type}"')
