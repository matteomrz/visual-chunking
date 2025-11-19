import os
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse import ResultType

from parsing.methods.config import Parsers
from parsing.model.document_parser import DocumentParser
from parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingResult,
)


class LlamaParseParser(DocumentParser):
    """Uses the LlamaParse API for parsing PDF documents"""

    module = Parsers.LLAMA_PARSE

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

        print(f"Successfully initialized Document Parser using LlamaParse")

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        raw_result = self.parser.parse(str(file_path))
        return raw_result.get_json()

    def _transform(self, raw_result: dict) -> ParsingResult:
        pages = raw_result.get("pages", [])

        root = ParsingResult.root()
        levels: list[ParsingResult | None] = [root]

        for page in pages:
            for idx, element in enumerate(page["items"]):
                type_label = element.get("type", "Unknown")
                height = page.get("height", 0)
                width = page.get("width", 0)

                json_bbox = element.get("bBox", {})
                x = json_bbox.get("x", 0.0)
                y = json_bbox.get("y", 0.0)
                w = json_bbox.get("w", 0.0)
                h = json_bbox.get("h", 0.0)

                # Fix the wrongly rotated coordinates -- TODO: Different Angles?
                if page.get("originalOrientationAngle", 0) != 0:
                    x = width - x - w
                    y = height - y - h

                bbox = ParsingBoundingBox(page=page, x=x, y=y, width=w, height=h)

                lvl = element.get("lvl")
                transformed = ParsingResult(
                    id=f"{page}_{idx}",
                    type=type_label,
                    content=element.get("md", ""),
                    geom=[bbox],
                )

                if lvl:
                    while len(levels) > lvl:
                        levels.pop()
                    for index in range(lvl):
                        if not levels[index]:
                            levels[index] = levels[-1]

                    levels[-1].children.append(transformed)
                    levels[lvl] = transformed
                else:
                    levels[-1].children.append(transformed)

        return root
