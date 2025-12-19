from typing import Any

from lib.parsing.methods.parsers import Parsers
from lib.parsing.methods.implementations.docling import DoclingParser
from lib.parsing.methods.implementations.llamaparse import LlamaParseParser
from lib.parsing.methods.implementations.mineru import MinerUParser
from lib.parsing.methods.implementations.unstructured import UnstructuredParser
from lib.parsing.methods.vlm import VLMParser
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
