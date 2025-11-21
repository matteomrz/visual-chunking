import json
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

from parsing.methods.config import Parsers
from parsing.model.document_parser import DocumentParser
from parsing.model.parsing_result import ParsingBoundingBox, ParsingResult

default_strat = "hi_res"
valid_strats = ["auto", "hi_res", "fast", "ocr_only"]


class UnstructuredParser(DocumentParser):
    module = Parsers.UNSTRUCTURED_IO

    def _parse(self, file_path: Path, options=None) -> dict:
        if not options:
            options = {}

        strat = options.get("strat", default_strat)

        if not strat in valid_strats:
            raise ValueError(f"""Error while creating UnstructuredParser: {strat} is not a valid strategy. 
                Valid strategies: {valid_strats}""")

        elements = partition_pdf(
            filename=file_path,
            strategy=strat,
            languages=["eng", "deu"],
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.image_path / file_path.stem,
        )

        json_str = elements_to_json(elements)
        return {"elements": json.loads(json_str)}

    def _transform(self, raw_result: dict) -> ParsingResult:
        root = ParsingResult.root()

        for element in raw_result.get("elements", []):
            metadata: dict = element.get("metadata", {})
            coordinates: dict = metadata.get("coordinates", {})
            points: list[list] = coordinates.get("points", [])
            page_width = coordinates.get("layout_width", 0.0)
            page_height = coordinates.get("layout_height", 0.0)

            if len(points) < 4:
                print(f"Malformed bounding box: {points}")
                continue

            if not page_width or not page_height:
                print(
                    f"Malformed page dimensions: height={page_height}, width={page_width}"
                )
                continue

            origin = points[0]
            end = points[2]

            if len(origin) < 2 or len(end) < 2:
                print(f"Malformed bounding box: {points}")
                continue

            l = origin[0] / page_width
            t = origin[1] / page_height
            r = end[0] / page_width
            b = end[1] / page_height

            b_box = ParsingBoundingBox(
                page=metadata.get("page_number", 0),
                left=l,
                top=t,
                right=r,
                bottom=b
            )

            transformed = ParsingResult(
                id=element.get("element_id", "unknown"),
                content=element.get("text", ""),
                type=element.get("type", "unknown"),
                geom=[b_box],
            )

            root.children.append(transformed)

        return root
