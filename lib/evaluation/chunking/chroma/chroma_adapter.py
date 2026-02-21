import logging
from pathlib import Path
from typing import Any, List

from chunking_evaluation import BaseChunker

from lib.evaluation.chunking.chroma.chroma_setup import CHROMA_DIR
from lib.parsing.model.parsing_result import ParsingResult, ParsingResultType
from lib.chunking.methods.chunkers import Chunkers
from lib.chunking.model.chunk import ChunkingResult
from lib.chunking.model.document_chunker import DocumentChunker
from lib.utils.thesis_names import get_chunker_name, get_chunker_param

logger = logging.getLogger(__name__)


def _transform_to_parsing_result(text: str) -> ParsingResult:
    """Transform plain text corpus into rudimentary ParsingResult."""

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

    @property
    def collection_name(self):
        module = self._inner_chunker.module
        max_tokens = self._inner_chunker.max_tokens
        param = 0
        match module:
            case Chunkers.FIXED_SIZE | Chunkers.RECURSIVE:
                if hasattr(self._inner_chunker, "overlap"):
                    param = self._inner_chunker.overlap
            case Chunkers.SEMANTIC:
                if hasattr(self._inner_chunker, "similarity_threshold_percentile"):
                    param = self._inner_chunker.similarity_threshold_percentile
            case Chunkers.HIERARCHICAL:
                if hasattr(self._inner_chunker, "max_parent_tokens"):
                    param = self._inner_chunker.max_parent_tokens

        return f"{module}_{max_tokens}_{param}"

    def get_info(self) -> dict[str, Any]:
        """Information about the chunking strategy and its parameters."""

        info: dict[str, Any] = {
            "Method": get_chunker_name(self._inner_chunker.module),
            "Param": get_chunker_param(self._inner_chunker),
            "N": self._inner_chunker.max_tokens
        }

        return info

    def split_text(self, text: str) -> List[str]:
        """Chunking adapter for general evaluation."""

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

        # Name of the directory of the outputs from the same parser
        batch_name = str(corpus.relative_to(CHROMA_DIR).with_suffix(""))

        try:
            results = self._inner_chunker.process_batch(batch_name, with_geom=False)
        except ValueError:
            logger.error(
                f"Missing ParsingResults for corpus {batch_name}. "
                f"Defaulting to string input."
            )
            return None

        chunks = []
        for result in results:
            chunks.extend(_transform_to_str_list(result))

        return chunks
