from typing import Any, Generator, Optional

from parsing.model.parsing_result import ParsingResult, ParsingResultType
from segmentation.methods.config import Chunkers
from segmentation.methods.recursive import find_rec_split
from segmentation.model.document_chunker import DocumentChunker
from segmentation.model.token import RichToken


class HierarchicalChunker(DocumentChunker):
    """Uses Docling's Hierarchical / Hybrid Chunking Strategy for document chunking."""

    module = Chunkers.HIERARCHICAL

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens if max_tokens else 128

    @classmethod
    def from_options(cls, options: dict):
        return cls(
            max_tokens=options.get("max_tokens", None)
        )

    @property
    def min_content_tokens(self):
        return self.max_tokens * 0.5

    def _get_chunk_tokens(self, document: ParsingResult) -> Generator[list[RichToken], Any, None]:
        yield from self._get_from_element(document, [])

    def _get_from_element(self, element: ParsingResult, parents: list[int]):
        parent_cnt = sum(parents)

        subtree_content = str(element)
        subtree_content_cnt = self._get_token_count(subtree_content)

        elem_tokens = self._encode(element)
        elem_token_cnt = len(elem_tokens)
        max_content_tokens = self.max_tokens - parent_cnt

        if subtree_content_cnt > max_content_tokens:
            # Go down one step further in the tree to perform the split
            if element.children:
                parents.append(elem_token_cnt)
                parent_cnt += elem_token_cnt

                while self.max_tokens - parent_cnt < self.min_content_tokens:
                    parent_cnt -= parents[0]
                    parents.pop(0)

                children_res: list[list[RichToken]] = []
                for child in element.children:
                    c_res = self._get_from_element(child, parents[:])
                    children_res.extend(c_res)

                # Find if there are any children which could be merged
                # Merge if they were not split at a lower stage and their length is ok
                i = 0
                while i < len(children_res) - 1:
                    c_1 = children_res[i]
                    c_2 = children_res[i + 1]

                    resulting_length = len(c_1) + len(c_2) + parent_cnt
                    if resulting_length < self.max_tokens:
                        c_1.extend(c_2)
                        children_res.pop(i + 1)
                    else:
                        i += 1

                for c_res in children_res:
                    if len(c_res) + elem_token_cnt < self.max_tokens:
                        c_res[0:0] = elem_tokens
                    yield c_res

            # Split the current Element content
            else:
                splits = find_rec_split(elem_tokens, max_content_tokens)
                last_split = 0
                for split in splits:
                    yield elem_tokens[last_split: split]
                    last_split = split

        else:
            if element.type != ParsingResultType.TABLE_ROW:
                for child in element.children:
                    child_tokens = self._encode(child)
                    elem_tokens.extend(child_tokens)

            yield elem_tokens
