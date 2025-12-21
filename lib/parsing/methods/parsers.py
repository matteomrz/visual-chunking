from enum import Enum


class Parsers(Enum):
    """Available methods for parsing PDF files to a structured data format."""

    LLAMA_PARSE = "llamaparse"
    UNSTRUCTURED_IO = "unstructured_io"
    DOCLING = "docling"
    # PADDLEOCR = "paddle_ocr"
    MINERU_PIPELINE = "mineru_pipeline"
    MINERU_VLM = "mineru_vlm"
    GEMINI = "gemini"

    @classmethod
    def default(cls):
        return cls.DOCLING

    @classmethod
    def get_parser_type(cls, name: str):
        return cls(name)
