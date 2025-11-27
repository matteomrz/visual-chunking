from parsing.model.parsing_result import ParsingResult
from segmentation.methods.config import Chunkers
from segmentation.model.chunk import Chunk, ChunkingResult
from segmentation.model.document_chunker import DocumentChunker


class FixedSizeChunker(DocumentChunker):
    max_tokens: int
    module = Chunkers.FIXED_SIZE

    def __init__(self, options: dict = None):
        if options:
            self.max_tokens = options.get("max_tokens", 128)
        else:
            self.max_tokens = 128

    def _segment(self, document: ParsingResult, options: dict = None) -> ChunkingResult:
        result = ChunkingResult(metadata=document.metadata)

        counter = 0
        tokens = []
        res_length = self.max_tokens

        for elem in _flatten(document):
            geom = elem.geom
            elem_tokens = self._encode(elem.content)
            bbox_tokens = [(geom, t) for t in elem_tokens]
            tokens.extend(bbox_tokens)

            while len(tokens) > res_length:
                chunk_tokens = tokens[:res_length]

                chunk = self._get_chunk(chunk_tokens, counter)
                result.chunks.append(chunk)
                counter += 1

                tokens = tokens[res_length:]
                res_length = self.max_tokens

        return result

    def _get_chunk(self, buffer_slice, idx: int) -> Chunk:
        content = self._decode([t[1] for t in buffer_slice])
        bounding_boxes = [t[0] for t in buffer_slice]
        unique = []
        for geom in bounding_boxes:
            for b in geom:
                if b not in unique:
                    unique.append(b)

        return Chunk(id=f"c_{idx}", content=content, geom=unique)


def _flatten(root: ParsingResult):
    for elem in root.children:
        yield elem
        yield from _flatten(elem)
