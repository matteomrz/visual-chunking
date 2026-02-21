from lib.parsing.methods.parsers import Parsers
from lib.chunking.methods.chunkers import Chunkers
from lib.chunking.model.document_chunker import DocumentChunker

_parser_latex_mapping = {
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

    return _parser_latex_mapping.get(parser, parser.value)


def get_is_parser_api(parser: str) -> str:
    return parser + ("*" if parser in _api_parsers else "")


_chunker_latex_mapping = {
    Chunkers.FIXED_SIZE: "Fixed-Size",
    Chunkers.RECURSIVE: "Recursive",
    Chunkers.SEMANTIC: "Semantic",
    Chunkers.HIERARCHICAL: "Hierarchical"
}


def get_chunker_name(chunker: Chunkers) -> str:
    """Returns a latex formatted string of the chunkers name."""
    return _chunker_latex_mapping.get(chunker, chunker.value)


def get_chunker_param(chunker: DocumentChunker) -> str:
    """
    Returns a latex formatted string of the chunkers secondary parameter. \
    (e.g. overlap, quantile, max_header_percentage)
    """

    module = chunker.module

    match module:
        case Chunkers.FIXED_SIZE | Chunkers.RECURSIVE:
            o = vars(chunker).get("overlap", 0)
            return f"$O={o}$"
        case Chunkers.SEMANTIC:
            q = vars(chunker).get("similarity_threshold_percentile", 0)
            return f"$Q={q}$"
        case Chunkers.HIERARCHICAL:
            b = vars(chunker).get("max_parent_tokens", 0)
            return f"$B_h={b}$"
