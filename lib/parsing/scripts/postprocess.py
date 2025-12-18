from pathlib import Path

from lib.parsing.model.parsing_result import (
    ParsingResult,
    ParsingResultType,
    ParsingMetaData as PmD,
)
from lib.parsing.scripts.spans import add_span_boxes

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

# Types where content == "" is acceptable
no_content_types = [
    ParsingResultType.ROOT,
    # FLOATING ITEMS
    ParsingResultType.FIGURE,
    ParsingResultType.TABLE,  # Content is stored in the Rows / Cells
    ParsingResultType.DOC_INDEX,
    # GROUP ITEMS
    ParsingResultType.LIST,
    ParsingResultType.REFERENCE_LIST
]


def _should_remove(element: ParsingResult, types_to_remove: list[ParsingResultType]) -> bool:
    """Checks whether an element should be removed."""
    is_unneeded = element.type in types_to_remove
    needs_content = element.type not in no_content_types
    is_empty = needs_content and element.content.strip() == ""
    return is_unneeded or is_empty


def _filter_elements(result: ParsingResult, types_to_remove: list[ParsingResultType]) -> bool:
    """Filter the parsed elements"""
    is_root = result.type == ParsingResultType.ROOT
    if not is_root and _should_remove(result, types_to_remove):
        return False

    filtered = []

    for child in result.children:
        if _filter_elements(child, types_to_remove):
            filtered.append(child)

    result.children = filtered
    return True


def _infer_hierarchy(root: ParsingResult):
    """
    This logic is adapted from Docling's Hierarchical Parser.
    After parsing most parsers do not show the document in a hierarchical order and instead as a flat list of elements.
    We use the documents reading order to infer which headers belong to which texts and restore the hierarchical ordering
    """

    # some parsers provide levels for the parsed headers
    # we keep a running list of these headers (e.g. [title-header, section-header, subsection-header...])
    # if we encounter a new header, we insert it into the list at the correct level
    # and remove all previous headers of higher levels (e.g. higher granularity)
    level_headings = [root]

    # copy the children list for iteration, so we can modify the original in the loop
    children = [c for c in root.children]

    for child in children:
        if child.type == ParsingResultType.SECTION_HEADER:
            lvl = child.metadata.get(PmD.HEADER_LEVEL.value, 1)

            while len(level_headings) > lvl:
                level_headings.pop()

            highest_lvl = len(level_headings)
            for index in range(lvl - highest_lvl):
                level_headings.append(level_headings[-1])

            parent = level_headings[-1]
            level_headings.append(child)

        else:
            parent = level_headings[-1]

        if parent.type != ParsingResultType.ROOT:
            child.parent = parent
            root.children.remove(child)
            parent.children.append(child)


def parse_post_process(file_path: Path, result: ParsingResult):
    """
    Perform post-processing after the parsing stage.
    Stages:
    Element filtering;
    Hierarchy inferring;
    Span bounding box creation;

    Args:
        file_path: Path to the parsed PDF guideline
        result: Output from the DocumentParser
    """
    _filter_elements(result, unneeded_types)
    _infer_hierarchy(result)
    add_span_boxes(file_path, result)
