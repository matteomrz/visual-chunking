from typing import Optional

from parsing.model.parsing_result import ParsingResult, ParsingResultType
from segmentation.methods.config import Chunkers
from segmentation.methods.recursive import find_rec_split
from segmentation.model.chunk import ChunkingResult
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

    def _handle_element(
        self,
        element: ParsingResult,
        parents: list[int],
        elem_info: dict,
    ) -> list[list[RichToken]]:
        # For each level we check if combining our children with us and our header fits in one chunk. If it does -> new chunk
        # If not go one level deeper. If we find a level where we can keep all of our children, check if your next door neighbor can also fit

        # While the content of the parents is longer than max_tokens, delete the highest header
        # Docling removes all
        min_content_token = self.max_tokens * 0.5
        parent_cnt = sum(parents)

        result = []
        subtree_content = str(element)
        subtree_content = self._get_token_count(subtree_content)

        elem_tokens, info = self._encode(element)
        elem_info[element.id] = info

        max_content_tokens = self.max_tokens - parent_cnt

        # We cant include this entire subtree in one chunk
        if subtree_content > max_content_tokens:
            # Go down one step further in the tree to perform the split
            if element.children:
                elem_content = element.content + element.get_delimiter()
                elem_content_cnt = self._get_token_count(elem_content)

                parents.append(elem_content_cnt)
                parent_cnt += elem_content_cnt

                while self.max_tokens - parent_cnt < min_content_token:
                    parent_cnt -= parents[0]
                    parents.pop(0)

                children_res: list[list[RichToken]] = []
                for child in element.children:
                    c_res = self._handle_element(child, parents[:], elem_info)
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

                c_1 = children_res[-1]
                for idx in range(len(children_res) - 1).__reversed__():
                    c_0 = children_res[idx]
                    if len(c_0) + len(c_1) + parent_cnt < self.max_tokens:
                        c_0.extend(c_1)
                        children_res.remove(c_1)
                    c_1 = c_0

                for c_res in children_res:
                    c_res[0:0] = elem_tokens

                result.extend(children_res)

            # Split the current Element content
            else:
                splits = find_rec_split(elem_tokens, max_content_tokens)
                last_split = 0
                for split in splits:
                    result.append(elem_tokens[last_split: split])
                    last_split = split
        else:
            if element.type != ParsingResultType.TABLE_ROW:
                for child in element.children:
                    child_tokens, child_info = self._encode(child)
                    elem_info[child.id] = child_info
                    elem_tokens.extend(child_tokens)

            result.append(elem_tokens)

        return result

    def segment(self, document: ParsingResult, with_geom: bool = True) -> ChunkingResult:
        result = ChunkingResult(metadata=document.metadata)
        elem_info = {}
        document.add_delimiters()
        chunk_tokens = self._handle_element(document, [], elem_info)

        for idx, token_list in enumerate(chunk_tokens):
            chunk = self.get_chunk(token_list, idx, elem_info, with_geom)
            result.chunks.append(chunk)

        return result
