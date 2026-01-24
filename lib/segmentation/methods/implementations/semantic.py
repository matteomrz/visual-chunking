from typing import Any, Generator, Optional

from typing import TypedDict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.parsing.model.parsing_result import ParsingResult
from lib.segmentation.methods.chunkers import Chunkers
from lib.segmentation.methods.implementations.recursive import find_splits
from lib.segmentation.model.document_chunker import DocumentChunker
from lib.segmentation.model.token import RichToken
from lib.utils.get_sentences import get_sentences, setup_nltk


class Sentence(TypedDict):
    tokens: list[RichToken]
    distance: torch.Tensor


class SemanticChunker(DocumentChunker):
    """Finds breakpoints based on the semantic similarity between neighboring sentences."""

    module = Chunkers.SEMANTIC

    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_threshold_percentile: int

    min_tokens: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold_percentile = kwargs.get("similarity_threshold", 95)
        self.min_tokens = kwargs.get("min_tokens", 0)

    def _get_chunk_tokens(self, document: ParsingResult) -> Generator[list[RichToken], Any, None]:
        # Import NLTK packages needed for sentence splitting
        setup_nltk()

        prev_embedding: Optional[torch.Tensor] = None
        sentences: list[Sentence] = []
        similarities = []

        for element in document.flatten():
            if element.type in self.excluded_types:
                continue

            if element.content == "":
                continue

            sentence_tokens = self._get_sentence_tokens(element)

            for tokens in sentence_tokens:
                text = "".join([t.text for t in tokens])
                embedding = self.embedding_model.encode_document(text, normalize_embeddings=True)

                if prev_embedding is None:
                    distance = torch.zeros(1, 1)
                else:
                    similarity = self.embedding_model.similarity(prev_embedding, embedding)
                    similarities.append(similarity)
                    distance = 1 - similarity

                sentence: Sentence = {
                    "tokens": tokens,
                    "distance": distance
                }

                sentences.append(sentence)
                prev_embedding = embedding

        distances = [s["distance"] for s in sentences]
        distance_breakpoint = np.percentile(distances, q=self.similarity_threshold_percentile)

        prev_break = 0
        chunk_tokens = []
        for idx, distance in enumerate(distances):
            if distance < distance_breakpoint and idx < len(distances) - 1:
                continue

            slice_tokens = [t for s in sentences[prev_break:idx] for t in s["tokens"]]

            curr_len = len(chunk_tokens)
            merge_len = curr_len + len(slice_tokens)

            # We cant merge and Chunk is already big enough
            if merge_len > self.max_tokens and curr_len >= self.min_tokens and curr_len > 0:

                # Current Chunk is well sized
                if curr_len <= self.max_tokens:
                    yield chunk_tokens
                    chunk_tokens.clear()

                # We need to split more granular -> RecursiveChunking
                else:
                    splits = find_splits(chunk_tokens, self.max_tokens)
                    splits.pop()
                    last_split = 0
                    for split in splits:
                        yield chunk_tokens[last_split:split]
                        last_split = split

                    chunk_tokens = chunk_tokens[last_split:]

            chunk_tokens.extend(slice_tokens)
            prev_break = idx

    def _get_sentence_tokens(self, element: ParsingResult) -> list[list[RichToken]]:
        sentences = get_sentences(element.content)
        tokens = self._tokenize(element)

        result = [[]]
        s_idx = 0

        for token in tokens:
            if token.text not in sentences[s_idx]:
                result.append([])
                s_idx += 1

            result[s_idx].append(token)

        return result
