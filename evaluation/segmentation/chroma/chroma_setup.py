import math
from pathlib import Path

from chunking_evaluation import SyntheticEvaluation
from dotenv import load_dotenv

from config import CONFIG_DIR, GUIDELINES_DIR
from parsing.methods.config import Parsers
from parsing.model.options import ParserOptions
from parsing.model.parsing_result import ParsingMetaData as PmD, ParsingResult
from parsing.scripts.get_parser import get_document_parser

CHROMA_DIR = CONFIG_DIR / "chroma"


def _create_chroma_text_file(result: ParsingResult, exist_ok: bool) -> str:
    """
    Translates the ParsingResult into a text file that can be used for query generation.
    Performs additional element filtering to remove tables and figures.
    """
    file_path = Path(result.metadata[PmD.GUIDELINE_PATH.value])
    parser = result.metadata[PmD.PARSER.value]

    rel_path = file_path.relative_to(GUIDELINES_DIR)
    txt_path = (CHROMA_DIR / parser / rel_path).with_suffix(".txt")

    if not (exist_ok and txt_path.exists()):
        chroma_str = str(result)
        with open(txt_path, "w") as f:
            f.write(chroma_str)

    return str(txt_path)


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

    options = {ParserOptions.EXIST_OK: parse_exist_ok}
    documents = doc_parser.process_batch(batch_name, options)

    output_dir = CHROMA_DIR / parser_name / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Info: Scanning {str(output_dir)} for text documents...")

    # Paths to .txt files that will be used for query generation
    output_paths = [
        _create_chroma_text_file(doc, parse_exist_ok)
        for doc in documents
    ]

    print(f"Info: Using {len(output_paths)} documents to create synthetic Chroma dataset...")

    load_dotenv()
    query_path = CHROMA_DIR / f"{batch_name}_{parser_name}.csv"
    skip_generation = query_exist_ok and query_path.exists()

    evaluation = SyntheticEvaluation(corpora_paths=output_paths, queries_csv_path=query_path)

    if not skip_generation:
        queries_per_file = math.ceil(question_count / len(output_paths))
        evaluation.generate_queries_and_excerpts(
            num_rounds=1,
            approximate_excerpts=True,
            queries_per_corpus=queries_per_file,
        )

    print("Success: Created synthetic evaluation!")
    return evaluation
