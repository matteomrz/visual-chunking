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
    ParsingResultType.FORM_AREA,
]

no_content_types = [
    ParsingResultType.ROOT,
    # FLOATING ITEMS
    ParsingResultType.FIGURE,
    ParsingResultType.TABLE,
    # GROUP ITEMS
    ParsingResultType.LIST,
    ParsingResultType.REFERENCE_LIST
]


def _should_remove(element: ParsingResult) -> bool:
    is_unneeded = element.type in unneeded_types
    needs_content = element.type not in no_content_types
    is_empty = needs_content and element.content.strip() == ""
    return is_unneeded or is_empty


def _filter_elements(result: ParsingResult) -> bool:
    is_root = result.type == ParsingResultType.ROOT
    if not is_root and _should_remove(result):
        return False

    filtered = []

    for child in result.children:
        if _filter_elements(child):
            filtered.append(child)

    result.children = filtered
    return True


def parse_post_process(file_path: Path, result: ParsingResult):
    _filter_elements(result)
    add_span_boxes(file_path, result)
