from pathlib import Path
from typing import List

from chunking_evaluation import BaseChunker

from config import BOUNDING_BOX_DIR
from evaluation.segmentation.chroma.chroma_setup import CHROMA_DIR
from parsing.model.parsing_result import ParsingResult, ParsingResultType
from segmentation.model.chunk import ChunkingResult
from segmentation.model.document_chunker import DocumentChunker


def _transform_to_parsing_result(text: str) -> ParsingResult:
    root = ParsingResult.root()
    # The text in the general evaluation is just plain text with newlines in between
    elements = text.split("\n\n")

    for idx, elem in enumerate(elements):
        res = ParsingResult(
            id=f"text_{idx}",
            content=elem,
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

    def get_info(self) -> dict:
        info = {"method": self._inner_chunker.__class__.__name__}
        if hasattr(self._inner_chunker, "max_tokens"):
            info["max_tokens"] = self._inner_chunker.__getattribute__("max_tokens")

        if hasattr(self._inner_chunker, "overlap"):
            info["overlap"] = self._inner_chunker.__getattribute__("overlap")

        return info

    def split_text(self, text: str) -> List[str]:
        doc = _transform_to_parsing_result(text)
        res = self._inner_chunker.segment(doc, with_geom=False)
        return _transform_to_str_list(res)

    def get_chunks_from_corpus_path(self, corpus_id: str) -> List[str] | None:
        """
        Overwrites a method added to the BaseChunker.
        During test execution, we first try to call this method before defaulting to ``split_text`` if None is returned.
        This allows us to get access to the ParsingResult when we evaluate on our Synthetic Dataset.
        """
        corpus = Path(corpus_id)
        is_synth = corpus.exists() and corpus.is_relative_to(CHROMA_DIR)
        if not is_synth:
            return None

        # Path to the output from the same parser - as a ParsingResult
        rel_path = corpus.relative_to(CHROMA_DIR)
        bbox_path = (BOUNDING_BOX_DIR / rel_path).with_suffix(".json")

        if not bbox_path.exists():
            print(
                f"Warning: Missing bounding boxes for corpus {corpus_id}. "
                f"Defaulting to string input."
            )
            return None

        result = self._inner_chunker.process_document(bbox_path, with_geom=False)
        return _transform_to_str_list(result)
