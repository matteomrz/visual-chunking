from enum import Enum


# available parsing methods

class Parsers(Enum):
    LLAMA_PARSE = "llamaparse"
    UNSTRUCTURED_IO = "unstructured_io"
    DOCLING = "docling"

    @classmethod
    def default(cls):
        return cls.DOCLING
