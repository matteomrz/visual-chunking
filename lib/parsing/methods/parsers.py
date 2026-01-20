from enum import Enum


class Parsers(Enum):
    """Available methods for parsing PDF files to a structured data format."""

    LLAMA_PARSE = "llamaparse"
    UNSTRUCTURED_IO = "unstructured_io"
    DOCLING = "docling"
    DOCLING_GRANITE = "docling_granite"
    # PADDLEOCR = "paddle_ocr"
    MINERU_PIPELINE = "mineru_pipeline"
    MINERU_VLM = "mineru_vlm"
    GEMINI = "gemini"
    DOCUMENT_AI = "document_ai"

    @classmethod
    def default(cls):
        return cls.DOCLING

    @classmethod
    def get_parser_type(cls, name: str):
        return cls(name)
