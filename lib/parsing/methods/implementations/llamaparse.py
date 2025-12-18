import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse import ResultType
from llama_cloud_services.parse.types import JobResult

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingResult,
    ParsingMetaData as PmD,
    ParsingResultType,
)


class LlamaParseParser(DocumentParser[JobResult]):
    """Uses the LlamaParse API for parsing PDF documents"""

    module = Parsers.LLAMA_PARSE

    label_mapping = {
        "heading": ParsingResultType.SECTION_HEADER,
        "text": ParsingResultType.PARAGRAPH,
        "table": ParsingResultType.TABLE,
    }

    def __init__(self):
        load_dotenv()
        key = os.getenv("LLAMAPARSE_API_KEY")

        if not key:
            raise ValueError(
                'LLamaParse API Key is not set correctly. Please set "LLAMAPARSE_API_KEY" in the .env file.'
            )

        self.parser = LlamaParse(
            api_key=key,
            result_type=ResultType.JSON,
            extract_layout=True,
            verbose=True,
        )

    def _parse(self, file_path: Path, options: dict = None) -> JobResult:
        raw_result = self.parser.parse(str(file_path))
        return raw_result

    def _transform(self, raw_result: JobResult) -> ParsingResult:
        json_result = raw_result.get_json()
        pages = json_result.get("pages", [])

        root = ParsingResult.root()

        for page_idx, page in enumerate(pages):
            height = page.get("height", 0)
            width = page.get("width", 0)
            layout_elems = page.get("layout", [])
            for idx, element in enumerate(page.get("items", [])):
                raw_type = element.get("type", "Unknown")
                type_label = self._get_element_type(raw_type)

                # Use the values from the layout section -> way more accurate
                l_elm = _find_first_and_remove(layout_elems, "label", raw_type)

                if l_elm:
                    bbox = l_elm.get("bbox", {})
                    l = bbox.get("x", 0.0)
                    t = bbox.get("y", 0.0)
                    r = bbox.get("w", 0.0) + l
                    b = bbox.get("h", 0.0) + t
                else:
                    json_bbox = element.get("bBox", {})
                    l = json_bbox.get("x", 0.0) / width
                    t = json_bbox.get("y", 0.0) / height
                    r = json_bbox.get("w", 0.0) / width + l
                    b = json_bbox.get("h", 0.0) / height + t

                # Fix the wrongly rotated coordinates -- TODO: Different Angles? Currently only 180 degrees
                if page.get("originalOrientationAngle", 0) != 0:
                    l = 1 - r
                    t = 1 - b

                bbox = ParsingBoundingBox(
                    page=page_idx + 1, left=l, top=t, right=r, bottom=b
                )

                # Find out if the element is a header and at which level
                transformed = ParsingResult(
                    id=f"p{page_idx}_{idx}",
                    type=type_label,
                    content=element.get("md", ""),
                    geom=[bbox],
                    parent=root,
                )

                lvl = element.get("lvl", -1)
                if lvl > 0:
                    transformed.metadata[PmD.HEADER_LEVEL.value] = lvl

                root.children.append(transformed)

        return root

    def _get_md(self, raw_result: JobResult, file_path: Path) -> str:
        return raw_result.get_markdown()


def _find_first_and_remove(values: list, key: str, val: str) -> Any:
    for elem in values:
        found = elem.get(key, "")
        if found and found == val:
            values.remove(elem)
            return elem

    return None
