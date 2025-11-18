from enum import Enum


# available parsing methods

class Parsers(Enum):
    LLAMA_PARSE = "llamaparse"
    UNSTRUCTURED_IO = "unstructured_io"
    DOCLING = "docling"

    @classmethod
    def default(cls):
        return cls.DOCLING

    @classmethod
    def get_parser(cls, name: str):
        for parser in cls:
            if parser.value == name:
                return parser

        raise ValueError(f'Parsing strategy "{name}" does not exist')
