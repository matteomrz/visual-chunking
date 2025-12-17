from typing import Any

from lib.parsing.methods.config import Parsers
from lib.parsing.methods.docling import DoclingParser
from lib.parsing.methods.llamaparse import LlamaParseParser
from lib.parsing.methods.mineru import MinerUParser
from lib.parsing.methods.unstructured import UnstructuredParser
from lib.parsing.model.document_parser import DocumentParser


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
