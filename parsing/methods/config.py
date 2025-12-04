from enum import Enum


class Parsers(Enum):
    """Available methods for parsing PDF files to a structured data format."""

    LLAMA_PARSE = "llamaparse"
    UNSTRUCTURED_IO = "unstructured_io"
    DOCLING = "docling"
    # PADDLEOCR = "paddle_ocr"
    MINERU_PIPELINE = "mineru_pipeline"
    MINERU_VLM = "mineru_vlm"

    @classmethod
    def default(cls):
        return cls.DOCLING

    @classmethod
    def get_parser(cls, name: str):
        for parser in cls:
            if parser.value == name:
                return parser

        raise ValueError(f'Parsing strategy "{name}" does not exist')
