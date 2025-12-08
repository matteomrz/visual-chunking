from pathlib import Path

from parsing.model.parsing_result import ParsingResult, ParsingResultType
from parsing.scripts.spans import add_span_boxes

unneeded_types = [
    # REFERENCES
    ParsingResultType.REFERENCE_LIST,
    ParsingResultType.REFERENCE_ITEM,
    # PAGE FURNITURE
    ParsingResultType.PAGE_FOOTER,
    ParsingResultType.PAGE_HEADER,
    # MISC
    ParsingResultType.FORM_AREA
]


def _filter_elements(result: ParsingResult):
    for element in result.flatten():
        if element.type in unneeded_types:
            element.parent.children.remove(element)


def parse_post_process(file_path: Path, result: ParsingResult):
    add_span_boxes(file_path, result)
