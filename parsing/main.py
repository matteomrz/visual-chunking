import argparse
from typing import Any

from config import DEFAULT_GUIDELINE
from parsing.methods.docling.parser import DoclingParser
from parsing.methods.llamaparse.parser import LlamaParseParser
from parsing.methods.config import Parsers
from parsing.methods.unstructured_io.parser import UnstructuredParser
from parsing.model.document_parser import DocumentParser

DEFAULT_PARSER = Parsers.default().value
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--draw", "-d", action="store_true", help="Create annotated PDF"
)
arg_parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=DEFAULT_GUIDELINE,
    help=f'PDF filename without extension. Default: "{DEFAULT_GUIDELINE}"',
)
arg_parser.add_argument(
    "--method",
    "-m",
    type=str,
    default=DEFAULT_PARSER,
    help=f'Supported PDF parsing method. Default: "{DEFAULT_PARSER}"',
    choices=[p.value for p in Parsers],
)

args = arg_parser.parse_args()


def _get_parser(parser_type: str, options: dict) -> DocumentParser[Any]:
    match parser_type:
        case Parsers.LLAMA_PARSE.value:
            return LlamaParseParser()
        case Parsers.DOCLING.value:
            return DoclingParser()
        case Parsers.UNSTRUCTURED_IO.value:
            return UnstructuredParser()
        case _:
            raise ValueError(f'No DocumentParser specified for type "{parser_type}"')


def main():
    parser = _get_parser(args.method, {})
    options = {}
    if args.draw:
        options["draw"] = True

    parser.process_document(file_name=args.file, options=options)


if __name__ == "__main__":
    main()
