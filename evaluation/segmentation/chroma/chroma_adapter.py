from typing import List

from chunking_evaluation import BaseChunker

from parsing.model.parsing_result import ParsingResult, ParsingResultType
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker


def _transform_to_parsing_result(text: str) -> ParsingResult:
    root = ParsingResult.root()
    elements = text.split("\n\n")

    for idx, elem in enumerate(elements):
        res = ParsingResult(
            id=f"text_{idx}",
            content=f"{elem}\n\n",
            type=ParsingResultType.PARAGRAPH,
            parent=root,
            geom=[],
        )
        root.children.append(res)

    return root


def _transform_to_str_list(result: ChunkingResult):
    return [c.content for c in result.chunks]


class ChromaChunker(BaseChunker):
    """Adapter to use a DocumentChunker for the Chroma chunking evaluation."""
    _inner_chunker = DocumentChunker

    def __init__(self, chunker: DocumentChunker):
        self._inner_chunker = chunker

    def split_text(self, text: str) -> List[str]:
        doc = _transform_to_parsing_result(text)
        res = self._inner_chunker.segment(doc, with_geom=False)
        return _transform_to_str_list(res)
