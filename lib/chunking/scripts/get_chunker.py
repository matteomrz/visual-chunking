from lib.chunking.methods.chunkers import Chunkers
from lib.chunking.model.document_chunker import DocumentChunker


def get_document_chunker(chunker_type: Chunkers, **kwargs) -> DocumentChunker:
    max_tokens = kwargs.get("max_tokens", None)

    match chunker_type:
        case Chunkers.FIXED_SIZE:
            from lib.chunking.methods.implementations.fixed_sized import FixedSizeChunker

            overlap = kwargs.get("overlap", None)
            return FixedSizeChunker(max_tokens=max_tokens, overlap=overlap)

        case Chunkers.RECURSIVE:
            from lib.chunking.methods.implementations.recursive import RecursiveChunker

            overlap = kwargs.get("overlap", None)
            return RecursiveChunker(max_tokens=max_tokens, overlap=overlap)

        case Chunkers.SEMANTIC:
            from lib.chunking.methods.implementations.semantic import SemanticChunker

            min_tokens = kwargs.get("min_tokens", None)
            similarity_percentile = kwargs.get("percentile", None)
            return SemanticChunker(
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                similarity_threshold=similarity_percentile
            )

        case Chunkers.HIERARCHICAL:
            from lib.chunking.methods.implementations.hierarchical import HierarchicalChunker

            heading_budget = kwargs.get("budget", None)
            return HierarchicalChunker(max_tokens=max_tokens, max_parent_tokens=heading_budget)

        case _:
            raise ValueError(f'No DocumentChunker specified for type "{chunker_type}"')
