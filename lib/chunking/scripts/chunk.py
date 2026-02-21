from config import PARSING_RESULT_DIR
from lib.chunking.methods.chunkers import Chunkers
from lib.chunking.scripts.get_chunker import get_document_chunker


def chunk_document(
    chunker_name: str,
    parser_name: str,
    src_name: str,
    is_batch: bool = False,
    draw: bool = False,
    **kwargs
):
    chunker_type = Chunkers.get_chunker_type(chunker_name)
    chunker = get_document_chunker(chunker_type, **kwargs)

    if is_batch:
        batch_name = f"{parser_name}/{src_name}"
        chunker.process_batch(batch_name, draw=draw)
    else:
        file_path = PARSING_RESULT_DIR / parser_name / f"{src_name}.json"
        chunker.process_document(file_path, draw=draw)
