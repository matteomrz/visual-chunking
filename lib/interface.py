from argparse import ArgumentParser, Namespace

from lib.chunking.scripts.chunk import chunk_document
from lib.parsing.methods.parsers import Parsers
from lib.parsing.scripts.parse import parse_pdf


def _add_chunking_arguments(parser: ArgumentParser):
    """
    Adds the possibility to specify a chunking strategy.
    If no strategy is selected, then no chunking will be performed.
    """

    # Every chunking strategy takes N as an argument
    base_chunker = ArgumentParser(add_help=False)
    base_chunker.add_argument("--max_tokens", "-N", type=int, required=False)

    chunking_parsers = parser.add_subparsers(
        dest="chunker",
        title="Chunking strategy",
        metavar="<chunker>",
        required=False,
        help="Chunking strategy for the document segmentation pipeline"
    )

    # FIXED SIZE
    fixed_size_parser = chunking_parsers.add_parser(
        "fixed_size", help="Fixed-size chunking", parents=[base_chunker]
    )
    fixed_size_parser.add_argument("--overlap", "-O", type=int, required=False)

    # RECURSIVE
    recursive_parser = chunking_parsers.add_parser(
        "recursive", help="Recursive character chunking", parents=[base_chunker]
    )
    recursive_parser.add_argument("--overlap", "-O", type=int, required=False)

    # SEMANTIC
    semantic_parser = chunking_parsers.add_parser(
        "semantic", help="Breakpoint-based semantic chunking", parents=[base_chunker]
    )
    semantic_parser.add_argument("--percentile", "-Q", type=int, required=False)
    semantic_parser.add_argument("--min_tokens", "-M", type=int, required=False)

    # HIERARCHICAL
    hierarchical_parser = chunking_parsers.add_parser(
        "hierarchical", help="Hierarchical chunking", parents=[base_chunker]
    )
    hierarchical_parser.add_argument("--budget", "-Bh", type=int, required=False)


def _make_dp_parsers(base_parser: ArgumentParser):
    """Adds a subparser for each document parsing implementation"""
    dp_parser = base_parser.add_subparsers(
        dest="parser",
        title="DP implementation",
        metavar="<parser>",
        required=True,
        help="Parsing implementation for the document segmentation pipeline"
    )

    for p in Parsers:
        p_parser = dp_parser.add_parser(p.value)
        _add_chunking_arguments(p_parser)


def _construct_parser() -> Namespace:
    base_parser = ArgumentParser(description="Document Segmentation Pipeline")
    _make_dp_parsers(base_parser)

    # INPUT FILE
    # input must either be a single file or a batch of files
    input_group = base_parser.add_mutually_exclusive_group()

    # SINGLE FILE INPUT
    input_group.add_argument(
        "--file", "-f", type=str,
        help=f'PDF filename without extension in "data/guidelines".',
    )

    # BATCH INPUT
    input_group.add_argument(
        "--batch", "-b", type=str,
        help=f'Directory containing a batch of PDF files in "data/guidelines".',
    )

    # PROCESSING FLAGS

    # EXIST_OK
    # Skips processing if the output already exists
    base_parser.add_argument(
        "--exist_ok", "-E", action="store_true",
        help="Skips parsing or chunking stages if there already exists an output for the strategy."
    )

    # DRAW
    # Produces an annotated document for both the chunking and parsing stage
    base_parser.add_argument(
        "--draw", "-D", action="store_true",
        help="Creates annotated PDF files showing the output of the parser and chunker.",
    )

    return base_parser.parse_args()


def interface():
    args = _construct_parser()

    if args.batch is not None:
        src_name = args.batch
        is_batch = True
    else:
        src_name = args.file
        is_batch = False

    parser = args.parser
    parse_pdf(
        parser_name=parser,
        src_name=src_name,
        is_batch=is_batch,
        draw=args.draw,
        exist_ok=args.exist_ok,
    )

    chunker = args.chunker
    if chunker:
        chunk_document(
            chunker_name=args.chunker,
            parser_name=args.parser,
            src_name=src_name,
            is_batch=is_batch,
            **vars(args)
        )


if __name__ == "__main__":
    interface()
