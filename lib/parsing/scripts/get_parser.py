from typing import Any

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser


def get_document_parser(parser_type: Parsers) -> DocumentParser[Any]:
    match parser_type:

        case Parsers.LLAMA_PARSE:
            from lib.parsing.methods.implementations.llamaparse import LlamaParseParser

            return LlamaParseParser()

        case Parsers.DOCLING:
            from lib.parsing.methods.implementations.docling import DoclingParser

            return DoclingParser(use_vlm=False)

        case Parsers.GRANITE_DOCLING:
            from lib.parsing.methods.implementations.docling import DoclingParser

            return DoclingParser(use_vlm=True)

        case Parsers.UNSTRUCTURED_IO:
            from lib.parsing.methods.implementations.unstructured import UnstructuredParser

            return UnstructuredParser()

        case Parsers.MINERU_PIPELINE:
            from lib.parsing.methods.implementations.mineru import MinerUParser

            return MinerUParser(use_vlm=False)

        case Parsers.MINERU_VLM:
            from lib.parsing.methods.implementations.mineru import MinerUParser

            return MinerUParser(use_vlm=True)

        case Parsers.GEMINI:
            from lib.parsing.methods.implementations.gemini import GeminiParser

            return GeminiParser()

        case Parsers.DOCUMENT_AI:
            from lib.parsing.methods.implementations.document_ai import DocumentAIParser

            return DocumentAIParser()

        case _:
            raise ValueError(f'No DocumentParser specified for type "{parser_type}"')
