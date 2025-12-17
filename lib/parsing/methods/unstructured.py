import logging
from pathlib import Path

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_md

from lib.parsing.methods.config import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingResult, ParsingResultType

logger = logging.getLogger(__name__)

default_strat = "hi_res"
valid_strats = ["auto", "hi_res", "fast", "ocr_only"]


class UnstructuredParser(DocumentParser[list[Element]]):
    module = Parsers.UNSTRUCTURED_IO

    label_mapping = {
        # TEXT
        "Title": ParsingResultType.HEADER,
        "NarrativeText": ParsingResultType.PARAGRAPH,
        # LIST
        "ListItem": ParsingResultType.LIST_ITEM,
        # FIGURES & TABLES
        "FigureCaption": ParsingResultType.CAPTION,
        "Image": ParsingResultType.FIGURE,
        "Table": ParsingResultType.TABLE,
        # MISC
        "Formula": ParsingResultType.FORMULA,
        "Footer": ParsingResultType.PAGE_FOOTER,
        "Header": ParsingResultType.PAGE_HEADER,
        "UncategorizedText": ParsingResultType.UNKNOWN,
    }

    def _parse(self, file_path: Path, options=None) -> list[Element]:
        if not options:
            options = {}

        strat = options.get("strat", default_strat)

        if not strat in valid_strats:
            raise ValueError(f"""Error while creating UnstructuredParser: {strat} is not a valid strategy. 
                Valid strategies: {valid_strats}""")

        image_dir = self._create_image_dir(file_path)
        elements = partition_pdf(
            filename=file_path,
            strategy=strat,
            languages=["eng", "deu"],
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=image_dir,
        )

        return elements

    def _transform(self, raw_result: list[Element]) -> ParsingResult:
        root = ParsingResult.root()

        for element in raw_result:
            metadata: dict = element.metadata.to_dict()
            coordinates: dict = metadata.get("coordinates", {})
            points: list[list] = coordinates.get("points", [])
            page_width = coordinates.get("layout_width", 0.0)
            page_height = coordinates.get("layout_height", 0.0)

            if len(points) < 4:
                logger.warning(f"Malformed bounding box: {points}")
                continue

            if not page_width or not page_height:
                logger.warning(f"Malformed page dimensions: h={page_height}, w={page_width}")
                continue

            origin = points[0]
            end = points[2]

            if len(origin) < 2 or len(end) < 2:
                logger.warning(f"Malformed bounding box: {points}")
                continue

            l = origin[0] / page_width
            t = origin[1] / page_height
            r = end[0] / page_width
            b = end[1] / page_height

            b_box = ParsingBoundingBox(
                page=metadata.get("page_number", 0), left=l, top=t, right=r, bottom=b
            )

            elem_type = self._get_element_type(element.category)

            transformed = ParsingResult(
                id=element.id,
                content=element.text,
                type=elem_type,
                parent=root,
                geom=[b_box],
            )

            root.children.append(transformed)

        return root

    def _get_md(self, raw_result: list[Element], file_path: Path) -> str:
        return elements_to_md(raw_result)
