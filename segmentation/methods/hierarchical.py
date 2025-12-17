from math import floor

from parsing.model.parsing_result import ParsingResult, ParsingResultType
from segmentation.methods.config import Chunkers
from segmentation.methods.recursive import find_splits
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import RichToken


class HierarchicalChunker(DocumentChunker):
    """Uses Docling's Hierarchical / Hybrid Chunking Strategy for document chunking."""

    module = Chunkers.HIERARCHICAL

    max_tokens: int
    # Limit how much of our chunk can be taken up by parent headers
    max_parent_tokens: int

    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.max_parent_tokens = floor(self.max_tokens * 0.5)

    def _get_chunk_tokens(self, document: ParsingResult):
        yield from self._get_from_element(document, [])

    def _get_from_element(self, element: ParsingResult, parent_tokens: list[int]):
        # While recursively iterating each element adds their token count to the end of parents
        # This way we always know how many elements are above us and how much space they need
        parent_cnt = sum(parent_tokens)
        max_content_tokens = self.max_tokens - parent_cnt

        # subtree includes all the children of the element
        # we always check if we have enough space to save the entire subtree
        subtree = str(element)
        subtree_cnt = self._get_token_count(subtree)

        # elem_tokens only have the tokens of the element itself
        elem_tokens = self._encode(element)
        elem_token_cnt = len(elem_tokens)

        if subtree_cnt > max_content_tokens:
            # Go down one step further in the tree to perform the split
            if element.children:
                parent_tokens.append(elem_token_cnt)
                parent_cnt += elem_token_cnt

                # If headers are too long, highest ones are removed
                while self.max_tokens - parent_cnt < self.max_parent_tokens:
                    parent_cnt -= parent_tokens[0]
                    parent_tokens.pop(0)

                # Two children can be merged if they were not merged in a lower stage
                # If a child was not split we keep it so we can merge the next child into it
                prev_tokens: list[RichToken] = []

                for child in element.children:
                    iterator = self._get_from_element(child, parent_tokens[:])

                    first_tokens = next(iterator)

                    # If our iterator returns any more elements we know that our child was split
                    for curr_tokens in iterator:
                        # Previous child and current child can not be merged
                        # Return tokens in the correct order

                        # Tokens from previous child
                        if prev_tokens:
                            yield self._add_parent_tokens(prev_tokens, elem_tokens)
                            prev_tokens = []

                        # Tokens from current child
                        if first_tokens:
                            yield self._add_parent_tokens(first_tokens, elem_tokens)
                            first_tokens = []

                        yield self._add_parent_tokens(curr_tokens, elem_tokens)

                    # Both the previous child and the current child were not split further
                    if prev_tokens and first_tokens:
                        resulting_length = len(prev_tokens) + len(first_tokens) + parent_cnt

                        if resulting_length <= self.max_tokens:
                            prev_tokens.extend(first_tokens)

                        else:
                            yield self._add_parent_tokens(prev_tokens, elem_tokens)
                            prev_tokens = first_tokens

                    else:
                        prev_tokens = first_tokens

            # element is a leaf node -> split content itself
            else:
                splits = find_splits(elem_tokens, max_content_tokens)
                last_split = 0
                for split in splits:
                    yield elem_tokens[last_split: split]
                    last_split = split

        # subtree fits in the same chunk
        else:
            if element.type != ParsingResultType.TABLE_ROW:
                for child in element.children:
                    child_tokens = self._encode(child)
                    elem_tokens.extend(child_tokens)

            yield elem_tokens

    def _add_parent_tokens(self, child: list, parent: list) -> list:
        if len(child) + len(parent) <= self.max_tokens:
            child[0:0] = parent
        return child
