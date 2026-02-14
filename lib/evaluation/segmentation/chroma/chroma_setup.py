import logging
import pandas as pd

from pathlib import Path

from chunking_evaluation import SyntheticEvaluation
from dotenv import load_dotenv

from config import CONFIG_DIR

from lib.evaluation.segmentation.chroma.load_annotations import get_offsets_path, load_annotations
from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.options import ParserOptions
from lib.parsing.model.parsing_result import ParsingMetaData

from lib.parsing.scripts.get_parser import get_document_parser

logger = logging.getLogger(__name__)

CHROMA_DIR = CONFIG_DIR / "chroma"


def _create_corpus(parser_type: Parsers, batch_name: str, exist_ok: bool) -> Path:
    doc_parser = get_document_parser(parser_type)
    parser_name = parser_type.value

    output_dir = CHROMA_DIR / parser_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = CHROMA_DIR / parser_name / f"{batch_name}.txt"

    # Offsets are needed to load medical QA pairs
    offsets_path = get_offsets_path(output_path)

    if exist_ok and output_path.exists():
        logger.info(
            f"Chroma document corpus exists at {str(output_path)}."
            "Skipping generation."
        )

    else:
        options = {ParserOptions.EXIST_OK: exist_ok}
        documents = doc_parser.process_batch(batch_name, options)

        logger.info(f"Using {len(documents)} documents to create synthetic Chroma evaluation...")

        offsets = []
        prev_offset = 0

        parts = []

        for doc in documents:
            doc_str = str(doc)
            pdf_name = Path(doc.metadata[ParsingMetaData.GUIDELINE_PATH.value]).stem

            parts.append(doc_str)

            # When we match the medical QA pairs,
            # we want to only search for their parsed counterparts in the correct document
            # Therefore we save where each document starts and ends
            end_offset = prev_offset + len(doc_str)
            offsets.append((pdf_name, prev_offset, end_offset - 1))
            prev_offset = end_offset

        # Create corpus that includes every document
        corpus = "\n\n".join(parts)

        with open(output_path, "w") as f:
            f.write(corpus)

        offset_df = pd.DataFrame(offsets, columns=["pdf_name", "start_index", "end_index"])
        offset_df.to_csv(offsets_path, index=False)

        logger.info(f"Finished creating chroma corpus at {output_path}.")

    return output_path


def setup_evaluation_from_medical_qas(
    parser_type: Parsers,
    qa_name: str,
    parse_exist_ok: bool = True,
    question_exist_ok: bool = True
):
    parser_name = parser_type.value
    corpus_path = _create_corpus(parser_type, qa_name, parse_exist_ok)
    qa_path = CHROMA_DIR / f"{qa_name}_{parser_name}.csv"

    if not (question_exist_ok and qa_path.exists()):
        anno_path = CHROMA_DIR / f"{qa_name}.jsonl"
        df = load_annotations(anno_path, corpus_path)
        df.to_csv(qa_path, index=False)

    load_dotenv()
    evaluation = SyntheticEvaluation(
        corpora_paths=[str(corpus_path)],
        queries_csv_path=str(qa_path),
        chroma_db_path=(CHROMA_DIR / parser_name / "db")
    )

    logger.info("Finished creating evaluation on the medical QA pairs.")
    return evaluation


def setup_synthetic_evaluation(
    parser_type: Parsers,
    batch_name: str,
    question_count: int = 25,
    parse_exist_ok: bool = True,
    query_exist_ok: bool = True,
) -> SyntheticEvaluation:
    """
    Sets up a synthetic evaluation on our guideline documents.
    Uses the Markdown output from the specified parser for the guidelines specified.

    Args:
        parser_type: The parser that is used to generate the markdown, skips parsing if output already exists
        batch_name: Directory name in ``GUIDELINE_DIR`` containing the guidelines
        question_count: How many questions should at least be generated for the evaluation (Will round up after dividing through file count)
        parse_exist_ok: Whether to skip document parsing if there already exists an output from this method. Default: True
        query_exist_ok: Whether to skip query generation if a csv file already exists for the batch. Default: True
    """
    parser_name = parser_type.value
    corpus_path = _create_corpus(parser_type, batch_name, parse_exist_ok)

    load_dotenv()
    query_path = CHROMA_DIR / f"{batch_name}_{parser_name}.csv"
    skip_generation = query_exist_ok and query_path.exists()

    evaluation = SyntheticEvaluation(
        corpora_paths=[str(corpus_path)],
        queries_csv_path=query_path,
        chroma_db_path=str(CHROMA_DIR / parser_name / "db")
    )

    if not skip_generation:
        evaluation.generate_queries_and_excerpts(
            num_rounds=1,
            approximate_excerpts=True,
            queries_per_corpus=question_count,
        )

    logger.info("Finished creating synthetic evaluation.")
    return evaluation
