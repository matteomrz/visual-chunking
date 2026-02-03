import logging
from abc import ABC
from pathlib import Path

from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox,
    ParsingMetaData as PmD,
    ParsingResult,
    ParsingResultType
)

logger = logging.getLogger(__name__)

md_filter_types = [
    ParsingResultType.PAGE_HEADER,
    ParsingResultType.PAGE_FOOTER,
    ParsingResultType.WATERMARK
]


class VLMParser(DocumentParser, ABC):
    """Base class that uses a single-stage VLM for document parsing."""

    label_mapping = {
        t.value: t
        for t in ParsingResultType
    }

    def _transform(self, raw_result: dict) -> ParsingResult:
        root = ParsingResult.root()

        seen_elements: dict[ParsingResultType, int] = {}
        for elem in raw_result.get("layout_elements", []):
            try:
                content = elem["content"]
                raw_type = elem["category"]

            except KeyError:
                logger.warning(f"Skipping malformed element: {elem}")
                continue

            elem_type = self._get_element_type(raw_type)

            type_cnt = seen_elements.get(elem_type, 0)
            elem_id = f"{elem_type.value}_{type_cnt}"
            seen_elements[elem_type] = type_cnt + 1

            raw_box = elem.get("bbox", {})
            try:
                page_num = raw_box["page_number"]
                points = raw_box["box_2d"]

                if len(points) != 4:
                    raise ValueError()

                # Coordinates from gemini are always 0-1000
                points = [p / 1000 for p in points]

                if any(p > 1.0 or p < 0.0 for p in points):
                    raise ValueError()

            except (KeyError, ValueError):
                logger.warning(f"Malformed bounding box: {raw_box}")
                points = [0.0, 0.0, 0.0, 0.0]
                page_num = 1

            box = ParsingBoundingBox(
                page=page_num,
                left=points[1],
                top=points[0],
                right=points[3],
                bottom=points[2]
            )

            res = ParsingResult(
                id=elem_id,
                type=elem_type,
                content=content,
                parent=root,
                geom=[box]
            )

            if elem_type == ParsingResultType.SECTION_HEADER:
                level_key = PmD.HEADER_LEVEL.value
                res.metadata[level_key] = elem.get(level_key, 1)

            root.children.append(res)

        return root
