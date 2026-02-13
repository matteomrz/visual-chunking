import logging
import math
from pathlib import Path

from chunking_evaluation import SyntheticEvaluation
from dotenv import load_dotenv

from config import CONFIG_DIR, GUIDELINES_DIR
from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.options import ParserOptions
from lib.parsing.model.parsing_result import ParsingMetaData as PmD, ParsingResult
from lib.parsing.scripts.get_parser import get_document_parser

logger = logging.getLogger(__name__)

CHROMA_DIR = CONFIG_DIR / "chroma"


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
    doc_parser = get_document_parser(parser_type)
    parser_name = parser_type.value

    output_dir = CHROMA_DIR / parser_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = CHROMA_DIR / parser_name / f"{batch_name}.txt"

    if parse_exist_ok and output_path.exists():
        logger.info(
            f"Chroma document corpus exists at {str(output_path)}."
            "Skipping generation."
        )

    else:
        options = {ParserOptions.EXIST_OK: parse_exist_ok}
        documents = doc_parser.process_batch(batch_name, options)

        logger.info(f"Using {len(documents)} documents to create synthetic Chroma evaluation...")

        corpus = "\n\n".join([str(doc) for doc in documents])

        with open(output_path, "w") as f:
            f.write(corpus)

        logger.info(f"Finished creating chroma corpus at {output_path}.")

    load_dotenv()
    query_path = CHROMA_DIR / f"{batch_name}_{parser_name}.csv"
    skip_generation = query_exist_ok and query_path.exists()

    evaluation = SyntheticEvaluation(
        corpora_paths=[output_path],
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
