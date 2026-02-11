from lib.parsing.methods.parsers import Parsers

_latex_mapping = {
    Parsers.LLAMA_PARSE: "LlamaParse",
    Parsers.UNSTRUCTURED_IO: "Unstructured.io",
    Parsers.DOCLING: "Docling",
    Parsers.GRANITE_DOCLING: "Granite Docling",
    Parsers.MINERU_PIPELINE: "MinerU 2.5 Pipeline",
    Parsers.MINERU_VLM: "MinerU 2.5 VLM",
    Parsers.GEMINI: "Gemini 2.5 Flash",
    Parsers.DOCUMENT_AI: "Document AI"
}

_api_parsers = [
    "LlamaParse",
    "Gemini 2.5 Flash",
    "Document AI"
]


def get_parser_thesis_name(parser: Parsers) -> str:
    """Returns a latex formatted string of the parsers name."""

    return _latex_mapping.get(parser, parser.value)


def get_is_parser_api(parser: str) -> str:
    return parser + ("*" if parser in _api_parsers else "")
